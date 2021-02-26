# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:04:12 2021

@author: Toby
"""
import math
import numpy as np
from keras import backend as back
from keras import models
from keras import layers
from keras import utils
import tensorflow as tf
from tensorflow.python.client import device_lib


import matplotlib.pyplot as plt


def standerdize(data):

    f = 0
    for n in data:
        mean = n.mean()
        ranges = max(n) - min(n)
        data[f] = ((n - mean) / ranges)
        f += 1

    return data



def splitData(data, yCount = 1, trainPercent = 0.9): # TODO: shuffle randomly USE traintestsplit https://www.youtube.com/watch?v=iMIWee_PXl8&ab_channel=TheSemicolon
    shape = data.shape
    
    trainTill = int(shape[0] * trainPercent)
    
    train = data[:trainTill, :]
    test = data[trainTill:, :]
    
    xTill = shape[1] - yCount
    
    train_X = train[:, :xTill]
    train_Y = train[:, xTill:]
    
    test_X = test[:, :xTill]
    test_Y = test[:, xTill:]
    
    trainshape = train_X.shape
    train_X = train_X.reshape(trainshape[0], trainshape[1], 1)
    train_Y = train_Y.reshape(trainshape[0], yCount)

    testshape = test_X.shape
    test_X = test_X.reshape(testshape[0], testshape[1], 1)
    test_Y = test_Y.reshape(testshape[0], yCount)
    
    return ((train_X, train_Y), (test_X, test_Y))

'''
print('GPU COUNT:', str(len(tf.config.experimental.list_physical_devices('GPU'))))
print(device_lib.list_local_devices())

tf.test.is_built_with_cuda()
print('GPU COUNT:', str(len(tf.config.experimental.list_physical_devices('GPU'))))
'''

#print(back.tensorflow_backend._get_available_gpus())

fileName = 'data.csv'
raw_data = open(fileName, 'rt')
data = np.loadtxt(raw_data, delimiter = ',', dtype = np.float)

data = data[:200, :]

outputCount = 50

data = standerdize(data)
(train_X, train_Y), (test_X, test_Y) = splitData(data, outputCount)

model = models.Sequential()

# input layer
model.add(layers.LSTM(48, input_shape = (None, 1), activation = 'relu'))

# hidden layers
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64))

# output layer
model.add(layers.Dense(outputCount, activation='linear')) # TODO: make output size 50


model.summary()


model.compile(loss='mean_squared_error', optimizer='adam'  , metrics=['accuracy'])
model.fit(train_X, train_Y, epochs=50, batch_size=64)

prediction = model.predict(test_X)

def plot(preCount, outputCount, test_X, test_Y, prediction, index):
    
    test_X = test_X[index]
    test_Y = test_Y[index]
    prediction = prediction[index]

    
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    
    ax.set(title = 'Predicted vs Actual', xlabel = 'sample', ylabel = 'value')
        
    
    indices = range(len(test_X) - preCount, len(test_X) + outputCount)
    postIndices = range(len(test_X), len(test_X) + outputCount)
    
    
    ax.plot(indices, np.concatenate((test_X[-preCount:].flatten(), test_Y.flatten())), color = 'blue')
    ax.plot(postIndices, prediction, color = 'orange')
    
    ax.axvline(x = len(test_X), color = 'green', linewidth = 2, linestyle = '--')

for i in range(len(test_X)):
    plot(outputCount, outputCount, test_X, test_Y, prediction, i)




# use this:
# https://www.youtube.com/watch?v=iMIWee_PXl8&ab_channel=StanfordUniversitySchoolofEngineering