#!/usr/bin/env python3

import logging
import numpy as np
import tensorflow as tf

from utils import uniform_sampler, binary_sampler
from data_loader import data_loader, data_sampler
from models import Discriminator, Generator

def gain(data, output, epochs, resume=None, batch_size=128, miss_rate=0.9, hint_rate=0.1, size=0, alpha=100):
    # Load data to npy
    ori_data_x = data_loader(data)
    ori_data_x, miss_data_x, data_m = data_sampler(ori_data_x, miss_rate, size)

    # Define mask matrix (0 denotes missing)
    data_m = 1 - np.isnan(miss_data_x)
    miss_data_x = np.nan_to_num(miss_data_x, 0)

    # Other parameters
    no, dim = miss_data_x.shape

    # Data preprocessing
    norm_data_batch = tf.data.Dataset.from_tensor_slices((ori_data_x, miss_data_x, data_m))
    norm_data_batch = norm_data_batch.shuffle(buffer_size=no)
    norm_data_batch = norm_data_batch.batch(batch_size)

    # Define model
    logging.info('resume:{}'.format(resume))
    try:
      discriminator = tf.keras.models.load_model("%s_D" % (resume))
      generator = tf.keras.models.load_model("%s_G" % (resume))
      logging.info("Load resume success!")
    except Exception as err:
      discriminator = Discriminator(int(dim))
      generator = Generator(int(dim))
      logging.info("Load resume fails [%s]", err)
      
    D_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    G_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    D_metric = tf.keras.metrics.BinaryAccuracy()
    G_metric = tf.keras.metrics.BinaryAccuracy()

    # Init Loss list
    D_loss_list = []
    G_loss_list = []
    D_acc_list = []
    G_acc_list = []
    
    logging.info("[MNIST] Training...")
    # Training loop
    for epoch in range(epochs):
        for step, (O_mb, X_mb, M_mb) in enumerate(norm_data_batch):

            X_mb = tf.cast(X_mb, dtype=tf.float32)
            M_mb = tf.cast(M_mb, dtype=tf.float32)
            Z_mb = uniform_sampler(0, 0.01, tf.shape(M_mb)[0], dim)

            X_mb = X_mb + (1 - M_mb) * Z_mb

            G_sample = generator([X_mb, M_mb], training=False)
            Hat_X = X_mb + G_sample * (1 - X_mb)

            H_mb_temp = binary_sampler(hint_rate, tf.shape(M_mb)[0], dim)
            H_mb = M_mb * tf.convert_to_tensor(H_mb_temp, dtype=np.float32)

            with tf.GradientTape() as tape:
                D_prob = discriminator([Hat_X, H_mb], training=True)
                D_loss = -tf.reduce_mean(M_mb * tf.math.log(D_prob + 1e-8) + (1 - M_mb)
                                        * tf.math.log(1. - D_prob + 1e-8))
            D_metric.update_state(M_mb, D_prob)
            D_acc = D_metric.result().numpy()

            grads = tape.gradient(D_loss, discriminator.trainable_weights)
            D_optimizer.apply_gradients(zip(grads, discriminator.trainable_weights))

            with tf.GradientTape() as tape:
                G_sample = generator([X_mb, M_mb], training=True)
                G_loss_temp = -tf.reduce_mean((1 - M_mb) * tf.math.log(D_prob + 1e-8))
                MSE_loss = tf.reduce_mean((M_mb * X_mb - M_mb * G_sample)**2) / tf.reduce_mean(M_mb)
                G_loss = G_loss_temp + alpha * MSE_loss
            G_metric.update_state(O_mb, G_sample)
            G_acc = G_metric.result().numpy()

            grads = tape.gradient(G_loss, generator.trainable_weights)
            G_optimizer.apply_gradients(zip(grads, generator.trainable_weights))

            # Recored Loss, Acc., Gradient
            D_loss_list.append(D_loss)
            G_loss_list.append(G_loss)
            D_acc_list.append(D_acc)
            G_acc_list.append(G_acc)

            # Verbose
            if step % 10 == 0:
                logging.info('Epoch:{:3d}\tSteps:{:3d}\tD_loss:{:.3g}\tG_loss:{:.3g}\tD_accuracy:{:.3g}\tG_accuracy:{:.3g}'.format(
                    epoch, step, D_loss, G_loss, D_acc, G_acc))
                    
    logging.info('Training Finish!!')
    # Save model
    # tf.saved_model.save(discriminator, '{}_D'.format(output))
    # tf.saved_model.save(generator, '{}_G'.format(output))

    discriminator.save("%s_D" % (output))
    generator.save("%s_G" % (output)) 

    metrics = dict()
    metrics['G_loss'] = G_loss
    metrics['G_acc'] = G_acc
    metrics['D_loss'] = D_loss
    metrics['D_acc'] = D_acc
    return metrics  
