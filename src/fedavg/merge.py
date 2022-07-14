#import torch
import logging
import tensorflow as tf
import numpy as np

def merge(models, merged_output_path,DorG):
    '''weights = [torch.load(m['path'], 'cpu') for m in models]
    total_data_size = sum(m['size'] for m in models)
    factors = [m['size'] / total_data_size for m in models]

    merged = {}
    for key in weights[0].keys():
        merged[key] = sum([w[key] * f for w, f in zip(weights, factors)])
        
    logging.debug("weights: [%s]",weights)
    logging.debug("merged: [%s]", merged)
    logging.debug("merged_output_path: [%s]", merged_output_path)
    torch.save(merged, merged_output_path)'''
    logging.info("Start aggregator!")
    #mode = ""
    if DorG == "D":
        logging.debug("This is Discriminator!!")
        discriminator = [tf.keras.models.load_model(m['path_D']) for m in models]
        weights = [w.get_weights() for w in discriminator]
        # logging.debug("D_Weights:", weights)
        # mode = "D"
        total_data_size = sum(m['size_D'] for m in models)
        factors = [m['size_D'] / total_data_size for m in models]
        # logging.debug('D_factors', factors)

    elif DorG == "G":
        logging.debug("This is Generator!!")
        generator = [tf.keras.models.load_model(m['path_G']) for m in models]
        weights = [w.get_weights() for w in generator]
        # logging.debug("G_Weights:", weights)
        # mode = "G"
        total_data_size = sum(m['size_G'] for m in models)
        factors = [m['size_G'] / total_data_size for m in models]
        # logging.debug('G_factors', factors)

    else:
        logging.debug('Error!!DorG has problem!!')

    logging.debug('weights_length[0]:{}'.format(len(weights[0])))
    weights = np.array(weights)
    logging.debug('weights.shape:',weights.shape)

    for i in range(len(weights)):
        if i == 0:
            merged = weights[i] + weights[i + 1]

    # logging.debug('merged shape:',len(merged))
    # logging.debug('merged shape:', len(merged[0]))
    # new_weights = []
    factors_weights = []
    for i in range(len(factors)):
        factors_weights.append(np.array(weights[i]) * factors[i])
        # for j in weights[i]:
        #     new_weights.append(j * factors[i])
        #     #logging.debug(len(new_weights))
        # factors_weights.append(new_weights)
    # logging.debug('factors_weights length:',len(factors_weights))
    # logging.debug('factors_weights length:', len(factors_weights[0]))
    # logging.debug('weights',weights[0][0])

    factors_weights = np.array(factors_weights)
    for i in range(len(factors_weights)):
        if i == 0:
            merged = factors_weights[i] + factors_weights[i + 1]
    # merged = factors_weights[0] + factors_weights[1]
    # logging.debug('factors_weights',factors_weights[0])
    # logging.debug('factors_weights',factors_weights[1][0])

    # logging.debug('merged:',merged.shape)
    # logging.debug('generator:',generator)
    # for i in range(0, len(new_weights)):
    #     if i == 0:
    #         merged.append(new_weights[i] + new_weights[i + 1])
    #         logging.debug('merged_length:',len(merged))
    #         i = i + 1
    #     else:
    #         merged.append(merged + new_weights[i])
    #         del merged[0:len(merged) // 2]
    #
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
    # discriminator = tf.keras.models.load_model('./test_weights1.tar_D').get_weights()
    # logging.debug('type:',type(discriminator))
    # logging.debug('discriminator:',discriminator)