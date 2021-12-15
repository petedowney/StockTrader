
import numpy as np
import matplotlib.pyplot as plt
import random


#
def isEven(num):
    if num == 0:
        return True
    elif abs(num) == 1:
        return False
    else:
        return isEven(abs(num) - 2)


def smallestIndex(array):
    smallest = 0
    smallest_array = array[0]
    for n in range(1, len(array)):
        if array[n] < smallest_array:
            smallest_array = array[n]
            smallest = n
    return smallest


def biggestIndex(array):
    biggest = 0
    biggest_array = array[0]
    for n in range(1, len(array)):
        if array[n] > biggest_array:
            biggest_array = array[n]
            biggest = n
    return biggest


def plotLoss(history):
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set(title='Loss vs. Time', xlabel='epoch', ylabel='loss')
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'], color='red')

def trunkate(data):

    for key in data.keys():
        data[key] = data[key][0]

    return data

def standerdize(data):
    meanList = []
    rangeList = []

    data2 = np.copy(data)

    for j, subD in enumerate(data):
        for i, n in enumerate(subD):
            mean = n.mean()
            ranges = n.max() - n.min()
            data2[j, i] = ((n - mean) / ranges)

            meanList.append(mean)
            rangeList.append(ranges)

        #data2[j] = np.column_stack(
        #    (data2[j], np.array(range(len(data2[j])))))  # indices are kept track of to match each row to its inverse


    #TODO fix mean and range list
    return data2, meanList, rangeList


def distributionPlotBefore(data):
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set(title="data Distribution Before Scaling", xlabel='Minutes', ylabel='Stock Value')

    for n in data:
        ax.plot(range(0, 15000, 15), n)

    plt.show()


def distributionPlotAfter(data):
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set(title="data Distribution After Scaling", xlabel='Minutes', ylabel='Stock Scaled Value')

    for n in data:
        ax.plot(range(0, 15000, 15), n[:-1])

    plt.show()


def splitData(data, y_count):
    shape = data.shape

    x_till = shape[2] - y_count



    X = np.array(data[:, :, :x_till])
    Y = np.array(data[:, :, x_till:])

    return (X, Y)


def plot(pre_count, output_count, test_x, test_y, prediction, index):
    test_x = test_x[index]
    test_y = test_y[index]
    prediction = prediction[index]

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set(title='Predicted vs Actual  (i=' + str(index) + ')', xlabel='sample', ylabel='value')
    indices = range(len(test_x) - pre_count, len(test_x) + output_count)
    post_indices = range(len(test_x), len(test_x) + output_count)
    ax.plot(indices, np.concatenate((test_x[-pre_count:].flatten(), test_y.flatten())), color='blue')
    ax.plot(post_indices, prediction, color='orange')
    ax.axvline(x=len(test_x), color='green', linewidth=2, linestyle='--')


def snipY(y):
    thisInv = []
    for val in y[:, :, -1]:
        thisInv.append(val)

    return y[:, :, :-1], thisInv

def randomSplit3rdD(x, y, splitpercent):

    x1 = None
    x2 = None
    y1 = None
    y2 = None

    len = x.shape[1]
    n1 = 0
    n2 = 0
    for n in range(0, len):

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
