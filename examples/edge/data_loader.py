import os
import gzip
import logging
import numpy as np

from utils import binary_sampler


def data_loader(data_name):
    """Loads datasets.
    Load varoius genotype file format to numpy array.
    The returned data has been transposed.
    One column represent one SNPs,
    One row represent one haplotype.

    Args:
        - data_name: relative path to data

    Returns:
        data_x: original data
    """
    filename, file_extension = os.path.splitext(data_name)

    if file_extension == '.gz':
        # Read with gzip function
        _, file_extension = os.path.splitext(filename)

        if file_extension == '.hap':
            # Read hap.gz file
            data_x = gzip.open(data_name, 'rt')
            data_x = np.genfromtxt(data_x, delimiter=' ')
            data_x = data_x.transpose()

        elif file_extension == '.gen':
            # Read gen.gz file
            data_x = []
            with gzip.open(data_name, 'rt') as f:
                for line in f:
                    line = line.split(" ")
                    line = line[5:]
                    data_x.append(line)

            data_x = np.array(data_x)
            data_x = data_x.transpose()

        else:
            print("Error: file format not supported.")
            exit()

    elif file_extension == '.hap':
        # Read hap file
        data_x = open(data_name, 'r')
        data_x = np.genfromtxt(data_x, delimiter=' ')
        data_x = data_x.transpose()

    elif file_extension == '.gen':
        # Read gen file
        data_x = []
        with open(data_name, 'r') as f:
            line = f.read()
            line = line.split(" ")
            line = line[5:]
            data_x.append(line)

        data_x = np.array(data_x)
        data_x = data_x.transpose()

    elif file_extension == '.npy':
        data_x = np.load(data_name).astype("float")
        data_x = data_x.transpose()

    else:
        logging.error('Error: file format not supported.')
        raise ValueError("File format not supported.")

    return data_x


def data_sampler(data_x, miss_rate=0.1, size=0):
    """Introduce missingness.

    Args:
        - data_x: data in np array
        - miss_rate: the probability of missing components

    Returns:
        data_x: original data
        miss_data_x: data with missing values
        data_m: indicator matrix for missing components
    """

    # Random select row
    if size != 0:
        no, dim = data_x.shape
        idx = np.random.randint(no, size=size)
        data_x = data_x[idx, :]

    # Parameters
    no, dim = data_x.shape

    # Introduce missing data
    data_m = binary_sampler(1 - miss_rate, no, dim)
    miss_data_x = data_x.copy()
    miss_data_x[data_m == 0] = np.nan

    return data_x, miss_data_x, data_m


def map_loader(map_name):
    data_map = np.loadtxt(map_name, delimiter=' ', skiprows=1)
    return data_map[:, 1]
