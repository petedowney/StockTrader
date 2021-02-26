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

def main():

    file_name = 'data.csv'
    raw_data = open(file_name, 'rt')
    data = np.loadtxt(raw_data, delimiter=',', dtype=np.float)

    data = data[:200, :]

    output_count = 30

    data = standerdize(data)
    (train_X, train_Y), (test_X, test_Y) = split_data(data, output_count)

    model = models.Sequential()
    
    # input layer
    model.add(layers.LSTM(48, input_shape=(None, 1), activation='swish'))

    # hidden layers
    model.add(layers.Dense(128, activation='swish'))
    model.add(layers.Dense(64, activation='linear'))

    # output layer
    model.add(layers.Dense(output_count, activation='linear'))  # TODO: make output size 50

    model.summary()

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.fit(train_X, train_Y, epochs=30, batch_size=64)

    prediction = model.predict(test_X)

    cost = (prediction - test_Y) ** 2
    avg_cost = [None] * len(cost[0])

    for x in range(0, len(cost[0])):
        avg_cost[x] = np.average(cost[:, x])

    temp = np.linspace(0, len(cost[0]) * 15, len(cost[0]))

    # why. why is this how you call it. who made this
    m, b = np.polyfit(temp, avg_cost, 1)

    print("Greatest Error:".rjust(18), "{a:.5}".format(a=np.max(cost)).rjust(10))
    print("Smallest Error:".rjust(18), "{a:.5}".format(a=np.min(cost)).rjust(10))
    print("Average Error:".rjust(18), "{a:.5}".format(a=np.average(cost)).rjust(10))
    print("Median:".rjust(18), "{a:.5}".format(a=np.median(cost)).rjust(10))
    print("STD:".rjust(18), "{a:.5}".format(a=np.std(cost)).rjust(10))
    print("Degradation Rate:".rjust(18), "{a:.5}".format(a=m * 15).rjust(10))

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.plot(range(0, len(cost[0]) * 15, 15), avg_cost)
    plt.plot(range(0, len(cost[0]) * 15, 15), m * temp + b)
    plt.show()


def standerdize(data):

    f = 0
    for n in data:
        mean = n.mean()
        ranges = max(n) - min(n)
        data[f] = ((n - mean) / ranges)
        f += 1

    return data


def split_data(data, y_count=1, train_percent=0.9):
    # TODO: shuffle randomly USE traintestsplit https://www.youtube.com/watch?v=iMIWee_PXl8&ab_channel=TheSemicolon
    shape = data.shape
    
    train_till = int(shape[0] * train_percent)
    
    train = data[:train_till, :]
    test = data[train_till:, :]
    
    x_till = shape[1] - y_count
    
    train_x = train[:, :x_till]
    train_y = train[:, x_till:]
    
    test_x = test[:, :x_till]
    test_y = test[:, x_till:]
    
    train_shape = train_x.shape
    train_x = train_x.reshape(train_shape[0], train_shape[1], 1)
    train_y = train_y.reshape(train_shape[0], y_count)

    test_shape = test_x.shape
    test_x = test_x.reshape(test_shape[0], test_shape[1], 1)
    test_y = test_y.reshape(test_shape[0], y_count)
    
    return (train_x, train_y), (test_x, test_y)


def plot(pre_count, output_count, test_x, test_y, prediction, index):
    
    test_x = test_x[index]
    test_y = test_y[index]
    prediction = prediction[index]

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set(title='Predicted vs Actual', xlabel='sample', ylabel = 'value')
    indices = range(len(test_x) - pre_count, len(test_x) + output_count)
    post_indices = range(len(test_x), len(test_x) + output_count)
    ax.plot(indices, np.concatenate((test_x[-pre_count:].flatten(), test_y.flatten())), color='blue')
    ax.plot(post_indices, prediction, color='orange')
    ax.axvline(x=len(test_x), color='green', linewidth=2, linestyle='--')

main()
