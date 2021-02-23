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



fileName = 'data.csv'
raw_data = open(fileName, 'rt')
data = np.loadtxt(raw_data, delimiter = ',', dtype = np.float)

def standerdize(data):

    print(len(data))

    f = 0
    for n in data:
        if np.std(n) != 0:
            mean = n.mean()
            ranges = max(n) - min(n)
            data[f] = ((n - mean) / ranges)
        f += 1

    return data



def splitData(data, yCount = 50, trainPercent = 0.9): # TODO: shuffle randomly USE traintestsplit https://www.youtube.com/watch?v=iMIWee_PXl8&ab_channel=TheSemicolon
    shape = data.shape
    
    trainTill = int(shape[0] * trainPercent)
    
    train = data[:trainTill, :]
    test = data[trainTill:, :]
    
    xTill = shape[1] - yCount
    
    train_X = train[:, :xTill]
    train_Y = train[:, xTill:]
    
    test_X = test[:, :xTill]
    test_Y = test[:, xTill:]
    
    return ((train_X, train_Y), (test_X, test_Y))

data = standerdize(data)
(train_X, train_Y), (test_X, test_Y) = splitData(data)

print(data)

model = models.Sequential()

#model.add(layers.Convolution1D(filters=10, kernel_size=5, padding='valid', activation='relu', input_shape=(1, 950,)))
#model.add(layers.AveragePooling2D(pool_size=2))

model.add(layers.Dense(120, activation='relu', input_shape=(950,)))
model.add(layers.Dense(1280, activation='relu'))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(50, activation='linear'))


model.summary()


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_X, train_Y, epochs=50, batch_size=64)

prediction = model.predict(train_X)


#print("Correct".rjust(23), "Incorrect".rjust(10))
print()

for x in range(0, len(prediction[0])):

    true = len(train_Y[x][abs(prediction[x] - train_Y[x]) <= 1]);
    false = len(train_Y[x][abs(prediction[x] - train_Y[x]) > 1]);

    accuracy = true / len(train_Y[x])

    print("Accuracy:".rjust(12), "{a:.3f}".format(a=accuracy).rjust(10), "row {a}".format(a=x + 1))




# use this:
# https://www.youtube.com/watch?v=iMIWee_PXl8&ab_channel=StanfordUniversitySchoolofEngineering