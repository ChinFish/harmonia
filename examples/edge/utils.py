"""Utility functions for GAIN.

(6) binary_sampler: sample binary random variables
(7) uniform_sampler: sample uniform random variables
(8) data_loader: data pre-processing, load data into .npy
"""

import numpy as np


def binary_sampler(p, rows, cols):
    """Sample binary random variables.

    Args:
      - p: probability of 1
      - rows: the number of rows
      - cols: the number of columns

    Returns:
      - binary_random_matrix: generated binary random matrix.
    """
    unif_random_matrix = np.random.uniform(0.0, 1.0, size=[rows, cols])
    binary_random_matrix = 1 * (unif_random_matrix > p)
    return binary_random_matrix


def uniform_sampler(low, high, rows, cols):
    """Sample uniform random variables.

    Args:
      - low: low limit
      - high: high limit
      - rows: the number of rows
      - cols: the number of columns

    Returns:
      - uniform_random_matrix: generated uniform random matrix.
    """
    return np.random.uniform(low, high, size=[rows, cols])


def rounding(imputed_data, data_x):
    """Round imputed data for categorical variables.

    Args:
      - imputed_data: imputed data
      - data_x: original data with missing values

    Returns:
      - rounded_data: rounded imputed data
    """

    _, dim = data_x.shape
    rounded_data = imputed_data.copy()

    for i in range(dim):
        temp = data_x[~np.isnan(data_x[:, i]), i]
        # Only for the categorical variable
        if len(np.unique(temp)) < 20:
            rounded_data[:, i] = np.round(rounded_data[:, i])

    return rounded_data


def normalization(data):
    '''Normalize data in [0, 1] range.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
    '''
    return data/2


def renormalization(data):
    return data*2
