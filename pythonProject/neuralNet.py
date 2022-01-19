
"""
Created on Mon Feb 22 15:04:12 2021

@author: Toby
@author: Pete Downey
"""
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt

#TODO
#blackman showles model

from sklearn.model_selection import train_test_split

from pythonProject import methods

# ASKTOBY
def NeuralNet():
    # DATA ===========
    file_name = 'data/techDataO.csv'
    raw_data = open(file_name, 'rt')
    dataO = np.loadtxt(raw_data, delimiter=',', dtype=np.float)

    file_name = 'data/techDataV.csv'
    raw_data = open(file_name, 'rt')
    dataV = np.loadtxt(raw_data, delimiter=',', dtype=np.float)


    # stacks the data into a single 3d array
    data = np.stack((dataO, dataV))

    outputCount = 50

    # splits up the data into the various types and seperates x and y and standerdizes
    test_X, test_Y, train_X, train_Y, val_X, val_Y = methods.fullStanderdize(data, outputCount)

    # gets rid of the volume data for y cause predictions only need to be x

    print(test_Y.shape)
    test_Y = methods.lossY(test_Y)
    print(test_Y.shape)
    val_Y = methods.lossY(val_Y)
    train_Y = methods.lossY(train_Y)


    # MODEL ========
    model = models.Sequential()

    # input layer (pre-network convolution)
    #model.add(layers.Conv2D(32, kernel_size=(1, 1),
    #                        # ((batch_size(None) channels (2)
    #                        # channels equates to the amount of data types each company has
    #                        input_shape=(None, None, 2),
    #                        activation='swish', padding="same"))

    model.add(layers.ConvLSTM1D(50, kernel_size=4,
                                input_shape=(1, 2, (1000-outputCount-1)),
                                activation='swish', return_sequences=False,
                                data_format='channels_first', padding="same"))

    #model.add(layers.AveragePooling2D(pool_size=(2,2)))

    # hidden layers
    model.add(layers.Dense(128, activation='swish'))
    model.add(layers.Dense(64, activation='linear'))

    # output layer
    model.add(layers.Dense(1, activation='linear'))

    model.summary()

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    history = model.fit(train_X, train_Y, epochs=30, batch_size=64, validation_data=(val_X, val_Y), verbose=1)

    methods.plotLoss(history)
    # code from:
    # https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    model_json = model.to_json()
    with open("data/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("data/model.h5")

    #print("Done")


if __name__ == "__main__":
    NeuralNet()
