
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

    data = np.stack((dataO, dataV))

    output_count = 50

    # standardization
    data, meanList, rangeList = methods.standerdize(data)

    X, Y = methods.splitData(data, output_count + 1)


    # split data
    train_X, test_X, train_Y, test_Y = methods.randomSplit3rdD(X, Y, 0.7)
    test_X, val_X, test_Y, val_Y = methods.randomSplit3rdD(test_X, test_Y, 0.5)


    #reshaped = lambda x: x.reshape(x.shape[0], x.shape[1], 1)

    #test_X = reshaped(test_X)
    #train_X = reshaped(train_X)
    #val_X = reshaped(val_X)

    pass

    test_Y, test_inverse = methods.snipY(test_Y)
    train_Y, train_inverse = methods.snipY(train_Y)
    val_Y, val_inverse = methods.snipY(val_Y)

    # MODEL ========
    model = models.Sequential()

    # input layer (pre-network convolution)
    model.add(layers.Conv2D(32, kernel_size=8, strides=(2, 2),
                            data_format="channels_first",
                            # ((batch_size(2,2) channels (2), (col row))
                            input_shape=(2, 2, 2),
                            activation='swish', padding="same"))

    model.add(layers.AveragePooling2D(data_format="channels_first", pool_size=(2,2)))

    ##input needs to change
    # LSTM
    #model.add(layers.LSTM(48, activation='swish', input_shape=(2, 2, None), return_sequences=False))

    # hidden layers
    model.add(layers.Dense(128, activation='swish'))
    model.add(layers.Dense(64, activation='linear'))

    # output layer
    model.add(layers.Dense(output_count, activation='linear'))

    #model.summary()

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    history = model.fit(train_X, train_Y, epochs=30, batch_size=64, validation_data=(val_X, val_Y), verbose=0)

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
