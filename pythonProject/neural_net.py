"""
Created on Mon Feb 22 15:04:12 2021

@author: Toby
@author: Pete Downey
"""
import numpy as np
import random
import copy
from keras import models
from keras import layers
from pythonProject import methods


def split_data(data, y_count):
    shape = data.shape

    X = np.array(data[:, :, :-y_count + 1])
    Y = np.array(data[:, :, -y_count + 1:])

    return (X, Y)


def standerdize(x, y):
    x_new = copy.deepcopy(x)
    y_new = copy.deepcopy(y)

    for j, subD in enumerate(x_new):
        for i, n in enumerate(subD):
            mean = n.mean()
            ranges = n.max() - n.min()
            x_new[j, i] = ((n - mean) / ranges)
            y_new[j, i] = ((y_new[j, i] - mean) / ranges)
        # data2[j] = np.column_stack(
        #    (data2[j], np.array(range(len(data2[j])))))  # indices are kept track of to match each row to its inverse

    # TODO fix mean and range list
    return x_new, y_new


# randomly splits x and y from the third dimension
# EX (10, 20, 30) -> (10, 20, 11) (10, 20, 19)
def random_split_3rdD(x, y, splitpercent):
    # TODO make this better
    x1 = None
    x2 = None
    y1 = None
    y2 = None

    n1 = 0
    n2 = 0
    for n in range(0, x.shape[1]):

        if random.random() < splitpercent:
            if n1 == 0:
                x1 = x[:, n, :]
                y1 = y[:, n, :]
            elif n1 == 1:
                x1 = np.stack((x1, x[:, n, :]), axis=1)
                y1 = np.stack((y1, y[:, n, :]), axis=1)
            else:
                x1 = np.hstack((x1, np.reshape(x[:, n, :], (x.shape[0], 1, x.shape[2]))))
                y1 = np.hstack((y1, np.reshape(y[:, n, :], (y.shape[0], 1, y.shape[2]))))
            n1 += 1
        else:
            if n2 == 0:
                x2 = x[:, n, :]
                y2 = y[:, n, :]
            elif n2 == 1:
                x2 = np.stack((x2, x[:, n, :]), axis=1)
                y2 = np.stack((y2, y[:, n, :]), axis=1)
            else:
                x2 = np.hstack((x2, np.reshape(x[:, n, :], (x.shape[0], 1, x.shape[2]))))
                y2 = np.hstack((y2, np.reshape(y[:, n, :], (y.shape[0], 1, y.shape[2]))))
            n2 += 1

    return x1, x2, y1, y2


def full_standerdize(data, outputCount):
    # splits data into x and y
    x, y = split_data(data, outputCount + 1)

    # standardization
    x, y = standerdize(x, y)

    # splits x and y into the various types
    train_x, test_x, train_y, test_y = random_split_3rdD(x, y, 0.7)
    test_x, val_x, test_y, val_y = random_split_3rdD(test_x, test_y, 0.5)

    # i am using lambdas solely because it makes toby happy

    # reformat the data to work with the NN
    # swaps axis 1 and 0 then adds an axis onto axis 1 creating a 4d array
    swap_and_expand = lambda data_input: np.expand_dims(np.swapaxes(data_input, 0, 1), axis=1)

    test_x = swap_and_expand(test_x)
    test_y = swap_and_expand(test_y)
    train_x = swap_and_expand(train_x)
    train_y = swap_and_expand(train_y)
    val_x = swap_and_expand(val_x)
    val_y = swap_and_expand(val_y)

    # converts the nn input format to its output format
    # gets rid of the volume data for y cause predictions only need to be x
    to_output_format = lambda data_input: np.swapaxes(np.squeeze(data_input[:, :, [0], :], axis=1), 1, 2)

    test_y = to_output_format(test_y)
    val_y = to_output_format(val_y)
    train_y = to_output_format(train_y)

    return test_x, test_y, train_x, train_y, val_x, val_y


# ASKTOBY
def train_neural_net():
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

    file_name = 'data/techDataEMA.csv'
    raw_data = open(file_name, 'rt')
    data_ema = np.loadtxt(raw_data, delimiter=',', dtype=np.float)

    # stacks the data into a single 3d array
    data = np.stack((data_o, data_v, data_r, data_ema))

    output_count = 50

    # splits up the data into the various types and seperates x and y and standerdizes
    test_x, test_y, train_x, train_y, val_x, val_y = full_standerdize(data, output_count)

    # honestly this is one of those things where the deeper you get the less it makes sense
    # the last output has an output of (node, outcount) ?????????????
    # don't even get me started on the convLSTM1D layer
    # MODEL ========
    model = models.Sequential()

    model.add(layers.ConvLSTM1D(50, kernel_size=20, strides=2,
                                input_shape=(1, 4, (1000 - output_count)),
                                activation='swish', return_sequences=False,
                                data_format='channels_first', padding="same"))

    model.add(layers.MaxPooling1D(pool_size=4, data_format='channels_first'))

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
    train_neural_net()
