#!/usr/bin/env python3

import logging
import numpy as np
import tensorflow as tf
import logging

from utils import uniform_sampler, binary_sampler
from data_loader import data_loader, data_sampler
from models import Discriminator, Generator


def gain(data, model_path, miss_rate=0.9, hint_rate=0.1, size=0, alpha=100):
    logging.debug('Begin merge test!!')
    # Load data to npy
    ori_data_x = data_loader(data)
    ori_data_x, miss_data_x, data_m = data_sampler(ori_data_x, miss_rate, size)

    # Define mask matrix (0 denotes missing)
    data_m = 1 - np.isnan(miss_data_x)
    miss_data_x = np.nan_to_num(miss_data_x, 0)

    # Other parameters
    no, dim = miss_data_x.shape

    # Define model
    discriminator = tf.keras.models.load_model("%s_D" % (model_path))
    generator = tf.keras.models.load_model("%s_G" % (model_path))

    D_metric = tf.keras.metrics.BinaryAccuracy()
    G_metric = tf.keras.metrics.BinaryAccuracy()

    # Testing
    O_mb = tf.cast(ori_data_x, dtype=tf.float32)
    X_mb = tf.cast(miss_data_x, dtype=tf.float32)
    M_mb = tf.cast(data_m, dtype=tf.float32)
    Z_mb = uniform_sampler(0, 0.01, tf.shape(M_mb)[0], dim)

    X_mb = X_mb + (1 - M_mb) * Z_mb

    G_sample = generator([X_mb, M_mb], training=False)
    Hat_X = X_mb + G_sample * (1 - X_mb)

    H_mb_temp = binary_sampler(hint_rate, tf.shape(M_mb)[0], dim)
    H_mb = M_mb * tf.convert_to_tensor(H_mb_temp, dtype=np.float32)

    D_prob = discriminator([Hat_X, H_mb], training=False)
    D_loss = -tf.reduce_mean(M_mb * tf.math.log(D_prob + 1e-8) + (1 - M_mb) * tf.math.log(1. - D_prob + 1e-8))
    D_metric.update_state(M_mb, D_prob)
    D_acc = D_metric.result().numpy()

    G_sample = generator([X_mb, M_mb], training=False)
    G_loss_temp = -tf.reduce_mean((1 - M_mb) * tf.math.log(D_prob + 1e-8))
    MSE_loss = tf.reduce_mean((M_mb * X_mb - M_mb * G_sample)**2) / tf.reduce_mean(M_mb)
    G_loss = G_loss_temp + alpha * MSE_loss
    G_metric.update_state(O_mb, G_sample)
    G_acc = G_metric.result().numpy()

    metrics = dict()
    logging.debug('Record metrics!!')
    metrics['G_loss'] = G_loss
    metrics['G_acc'] = G_acc
    metrics['D_loss'] = D_loss
    metrics['D_acc'] = D_acc
    logging.debug('Finish merge test!!')
    return metrics  