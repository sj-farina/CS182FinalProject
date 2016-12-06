import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections  as mc
import random as rd
import math, collections, sys
import csv


def loadData(file):
    file = np.genfromtxt(file, delimiter=',', skip_header=1,
            skip_footer=1, names=['date', 'open', 'high', 'low', 'close', 'adj'])
    open_value = file['open']
    close_value = file['close']
    # Zips the opening and closing values into one array
    return close_value


INFILE = 'BA_15Y_01_13.csv'
data_set = loadData(INFILE)
print len(data_set)
print data_set