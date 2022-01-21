
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
    data_o = np.loadtxt(raw_data, delimiter=',', dtype=np.float)

    file_name = 'data/techDataV.csv'
    raw_data = open(file_name, 'rt')
    data_v = np.loadtxt(raw_data, delimiter=',', dtype=np.float)

    file_name = 'data/techDataR.csv'
    raw_data = open(file_name, 'rt')
    data_r = np.loadtxt(raw_data, delimiter=',', dtype=np.float)


    # stacks the data into a single 3d array
    data = np.stack((data_o, data_v, data_r))

    output_count = 50

    # splits up the data into the various types and seperates x and y and standerdizes
    test_x, test_y, train_x, train_y, val_x, val_y = methods.fullStanderdize(data, output_count)

    # honestly this is one of those things where the deeper you get the less it makes sense
    # the last output has an output of (node, outcount) ?????????????
    # don't even get me started on the convLSTM1D layer
    # MODEL ========
    model = models.Sequential()

    model.add(layers.ConvLSTM1D(50, kernel_size=4,
                                input_shape=(1, 3, (1000-output_count)),
                                activation='swish', return_sequences=False,
                                data_format='channels_first', padding="same"))

    model.add(layers.AveragePooling1D(pool_size=4, data_format='channels_first'))

    # hidden layers
    model.add(layers.Dense(128, activation='swish'))
    model.add(layers.Dense(64, activation='swish'))

    # output layer
    model.add(layers.Dense(1, activation='linear'))

    model.summary()

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    # trains the model
    history = model.fit(train_x, train_y, epochs=30, batch_size=64, validation_data=(val_x, val_y), verbose=1)

    methods.plotLoss(history)

    # saves data
    # code from: https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    model_json = model.to_json()
    with open("data/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("data/model.h5")


if __name__ == "__main__":
    NeuralNet()
