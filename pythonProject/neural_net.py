# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:04:12 2021

@author: Toby
@author: Pete Downey
"""
import numpy as np
from keras import models
from keras import layers
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


fileName = 'data.csv'
raw_data = open(fileName, 'rt')
data = np.loadtxt(raw_data, delimiter = ',', dtype=np.float)

data = data[:200, :]

outputCount = 30

data = standerdize(data)
(train_X, train_Y), (test_X, test_Y) = splitData(data, outputCount)

model = models.Sequential()

# input layer
model.add(layers.LSTM(48, input_shape = (None, 1), activation='swish'))

# hidden layers
model.add(layers.Dense(128, activation='swish'))
model.add(layers.Dense(64, activation='linear'))

# output layer
model.add(layers.Dense(outputCount, activation='linear')) # TODO: make output size 50


model.summary()


model.compile(loss='mean_squared_error', optimizer='adam'  , metrics=['accuracy'])
model.fit(train_X, train_Y, epochs=30, batch_size=64)

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

#for i in range(len(test_X)):
#    plot(outputCount, outputCount, test_X, test_Y, prediction, i)
#toby you cant just open up 50 different plots the max is 20

cost = (prediction - test_Y)**2
avg_Cost = [None] * len(cost[0])

for x in range(0, len(cost[0])):
    avg_Cost[x] = np.average(cost[:, x])

temp = np.linspace(0, len(cost[0]) * 15, len(cost[0]))

# why. why is this how you call it. who made this
m, b = np.polyfit(temp, avg_Cost, 1)

print("Greatest Error:".rjust(18), "{a:.5}".format(a=np.max(cost)).rjust(10))
print("Smallest Error:".rjust(18), "{a:.5}".format(a=np.min(cost)).rjust(10))
print("Average Error:".rjust(18), "{a:.5}".format(a=np.average(cost)).rjust(10))
print("Median:".rjust(18), "{a:.5}".format(a=np.median(cost)).rjust(10))
print("STD:".rjust(18), "{a:.5}".format(a=np.std(cost)).rjust(10))
print("Degradation Rate:".rjust(18), "{a:.5}".format(a=m * 15).rjust(10))

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.plot(range(0, len(cost[0]) * 15, 15), avg_Cost)
plt.plot(range(0, len(cost[0]) * 15, 15), m * temp + b)
plt.show()