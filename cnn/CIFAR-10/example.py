#https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/examples/tutorials/layers/cnn_mnist.py# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pandas as pd
from numpy import array


# One-Dimensional List to Array
data = [11, 22, 33, 44, 55]
arr = array(data)

# Two-Dimensional List of Lists to Array
data = [[11, 22],
        [33, 44],
        [55, 66]
        ]
arr = array(data)
print(data)
print(type(data))
#print(data.shape)

print(arr)
print(type(arr))
print(arr.shape[0])


# define array
data = array([11, 22, 33, 44, 55])
print(data.shape)
data[0]
# reshape
data = data.reshape((data.shape[0], 1))
print(data.shape)
print(data.shape[0])
print(data.shape[1])

data3 = data.reshape((data.shape[0], data.shape[1], 1))

print(data3.shape)
print(data3.shape[0])
print(data3.shape[1])
print(data3.shape[2])

data2 = data3.reshape((5, 1))
data1 = data2.reshape(5)


data = np.arange(0,3072,1)
data = data.reshape(32, 32, 3)
print(data[0, 0, 0])
print(data[0, 0, 1])
print(data[0, 1, 0])