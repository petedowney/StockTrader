
import numpy as np
import matplotlib.pyplot as plt


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


def standerdize(data):
    meanList = []
    rangeList = []

    for i, n in enumerate(data):
        mean = n.mean()
        ranges = n.max() - n.min()
        data[i] = ((n - mean) / ranges)

        meanList.append(mean)
        rangeList.append(ranges)

    data = np.column_stack(
        (data, np.array(range(len(data)))))  # indices are kept track of to match each row to its inverse

    return data, meanList, rangeList


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

    x_till = shape[1] - y_count

    X = data[:, :x_till]
    Y = data[:, x_till:]

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
    for val in y[:, -1]:
        thisInv.append(int(val))

    return (y[:, :-1], thisInv)