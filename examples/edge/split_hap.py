#!/usr/bin/env python3

""" Split .hap file.
    
@version:1.0
@date:2022/08/03
"""

import numpy as np
from data_loader import data_loader

IN_FILE_PATH = ""
OUT_FILE_PATH = ""

# Read input file to numpy
# inFile IS TRANSPOSED!!!
# row represent a smaple
# column represent SNPs
inFile = data_loader(IN_FILE_PATH)

# Split data via numpy
outFile = inFile[0:1000]

# Traspose to .hap format
outFile = outFile.transpose()

# Save to txt
np.savetxt(OUT_FILE_PATH, outFile, fmt='%i', delimiter=' ')