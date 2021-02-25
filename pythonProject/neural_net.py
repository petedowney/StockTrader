# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:04:12 2021

@author: Toby
"""
import math
import numpy as np
from keras import models
from keras import layers
from keras import utils
import tensorflow


def standerdize(data):

    print(len(data))

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
    train_Y = train_Y.reshape(trainshape[0])

    testshape = test_X.shape
    test_X = test_X.reshape(testshape[0], testshape[1], 1)
    test_Y = test_Y.reshape(testshape[0])
    
    return ((train_X, train_Y), (test_X, test_Y))

fileName = 'data.csv'
raw_data = open(fileName, 'rt')
data = np.loadtxt(raw_data, delimiter = ',', dtype = np.float)


data = standerdize(data)
(train_X, train_Y), (test_X, test_Y) = splitData(data)



train_X = train_X.reshape(2363, 999, 1)
test_X = test_X.reshape(263, 999, 1)

train_Y = train_Y.reshape(2363)
test_Y = test_Y.reshape(263)

print(data)

model = models.Sequential()

# input layer
model.add(layers.LSTM(128, input_shape = (None, 1), activation = 'relu'))

# hidden layers
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32))

# output layer
model.add(layers.Dense(1, activation='linear')) # TODO: make output size 50


model.summary()


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_X, train_Y, epochs=20, batch_size=64)

prediction = model.predict(test_X)


#print("Correct".rjust(23), "Incorrect".rjust(10))
print()
t_Accuracy = 0
for x in range(0, len(prediction[0])):

    true = len(test_Y[x][abs(prediction[x] - test_Y[x]) <= 1]);
    false = len(test_Y[x][abs(prediction[x] - test_Y[x]) > 1]);

    accuracy = true / len(test_Y[x])
    t_Accuracy += accuracy;

    print("Accuracy:".rjust(12), "{a:.3f}".format(a=accuracy).rjust(10), "row {a}".format(a=x + 1))

t_Accuracy /= len(prediction[0])
print("Average Accuracy:".rjust(12), "{a:.3f}".format(a=t_Accuracy).rjust(10))


# use this:
# https://www.youtube.com/watch?v=iMIWee_PXl8&ab_channel=StanfordUniversitySchoolofEngineering