# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:04:12 2021

@author: Toby
"""

import numpy as np
from keras import models
from keras import layers



fileName = 'data.csv'
raw_data = open(fileName, 'rt')
data = np.loadtxt(raw_data, delimiter = ',', dtype = np.float)

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


(train_X, train_Y), (test_X, test_Y) = splitData(data)



model = models.Sequential()

model.add(layers.Dense(120, activation='relu', input_shape=(950,)))
model.add(layers.Dense(1280, activation='relu'))
model.add(layers.Dense(2480, activation='relu'))
model.add(layers.Dense(50, activation='linear'))


model.summary()


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_X, train_Y, epochs=50, batch_size=64)

# use this:
# https://www.youtube.com/watch?v=iMIWee_PXl8&ab_channel=StanfordUniversitySchoolofEngineering