#import torch
import logging
import tensorflow as tf
import numpy as np

def merge(models, merged_output_path,DorG):
    logging.info("Start aggregator!")
    
    if DorG == "D":
        logging.debug("This is Discriminator!!")
        discriminator = [tf.keras.models.load_model(m['path_D']) for m in models]
        weights = [w.get_weights() for w in discriminator]
        # logging.debug("D_Weights:", weights)
        total_data_size = sum(m['size_D'] for m in models)
        factors = [m['size_D'] / total_data_size for m in models]
        # logging.debug('D_factors', factors)

    elif DorG == "G":
        logging.debug("This is Generator!!")
        generator = [tf.keras.models.load_model(m['path_G']) for m in models]
        weights = [w.get_weights() for w in generator]
        # logging.debug("G_Weights:", weights)
        total_data_size = sum(m['size_G'] for m in models)
        factors = [m['size_G'] / total_data_size for m in models]
        # logging.debug('G_factors', factors)

    else:
        logging.debug('Error!!DorG has problem!!')

    #logging.debug('weights_length[0]:{}'.format(len(weights[0])))
    weights = np.array(weights)
    #logging.debug('weights.shape:',weights.shape)

<<<<<<< HEAD
=======
    #for i in range(len(weights)):
        #if i == 0:
            #merged = weights[i] + weights[i + 1]

>>>>>>> 3502c6d24b26a7b5e62ce3fd732e0d31a8c37125
    # logging.debug('merged shape:',len(merged))
    # logging.debug('merged shape:', len(merged[0]))
    factors_weights = []
    for i in range(len(factors)):
        factors_weights.append(np.array(weights[i]) * factors[i])
        # factors_weights.append(new_weights)
    # logging.debug('factors_weights length:',len(factors_weights))
    # logging.debug('factors_weights length:', len(factors_weights[0]))
    # logging.debug('weights',weights[0][0])

    factors_weights = np.array(factors_weights)
    merged = sum(factors_weights)
    # logging.debug('factors_weights',factors_weights[0])
    # logging.debug('factors_weights',factors_weights[1][0])

    # logging.debug('merged:',merged.shape)
    # logging.debug('generator:',generator)
    if DorG == "D":
        discriminator[0].set_weights(merged)
        discriminator[0].save("%s" % (merged_output_path))

    elif DorG == "G":
        generator[0].set_weights(merged)
        generator[0].save("%s" % (merged_output_path))

    else:
        logging.debug('Error!!Mode not D or G!!')
    # logging.debug('weights1:',weights[0])
    # logging.debug('weights2:',weights[1])
    # logging.debug('aggregate weights:',generator[0].get_weights())
    logging.debug('merge creating...')

    #calculate performance


    #return metrics

    # discriminator = tf.keras.models.load_model('./test_weights1.tar_D').get_weights()
    # logging.debug('type:',type(discriminator))
    # logging.debug('discriminator:',discriminator)