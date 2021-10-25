
"""
Created on Mon Feb 22 15:04:12 2021

@author: Toby
@author: Pete Downey
"""
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from pythonProject import methods

# ASKTOBY
def NeuralNet():
    # DATA ===========
    file_name = 'data/techData.csv'
    raw_data = open(file_name, 'rt')
    data = np.loadtxt(raw_data, delimiter=',', dtype=np.float)

    output_count = 50

    #methods.distributionPlotBefore(data)

    # standardization
    data, meanList, rangeList = methods.standerdize(data)

    #methods.distributionPlotAfter(data)

    X, Y = methods.splitData(data, output_count + 1)

    # split data
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.3)
    train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size=0.5)

    reshaped = lambda x: x.reshape(x.shape[0], x.shape[1], 1)

    test_X = reshaped(test_X)
    train_X = reshaped(train_X)
    val_X = reshaped(val_X)

    test_Y, test_inverse = methods.snipY(test_Y)
    train_Y, train_inverse = methods.snipY(train_Y)
    val_Y, val_inverse = methods.snipY(val_Y)

    # MODEL ========
    model = models.Sequential()

    # input layer (pre-network convolution)
    model.add(layers.Conv1D(32, kernel_size=8, strides=1, input_shape=(None, 1), activation='swish', padding="causal"))
    model.add(layers.AveragePooling1D(2))

    # LSTM
    model.add(layers.LSTM(48, activation='swish', input_shape=(None, 1), return_sequences=False))

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
