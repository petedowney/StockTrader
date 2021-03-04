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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
    
    invList = []
    
    for i, n in enumerate(data):
        mean = n.mean()
        ranges = n.max() - n.min()
        data[i] = ((n - mean) / ranges)
        
        invList.append(lambda x: x * ranges + mean)
    
    data = np.column_stack((data, np.array(range(len(data)))))
    
    '''
    mean = data.mean()
    diff = data.max() - data.min()
    data = (data - mean) / diff
    inverse = lambda x: x * diff + mean
    '''
    
    return data, invList


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

def snipY(y, invList):
    
    thisInv = []
    for val in y[:, -1]:
        thisInv.append(invList[int(val)])
    
    
    return (y[:, :-1], thisInv)

file_name = 'techData.csv'
raw_data = open(file_name, 'rt')
data = np.loadtxt(raw_data, delimiter=',', dtype=np.float)

output_count = 50

data, invList = standerdize(data)

X, Y = splitData(data, output_count + 1)

# split data
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.3)
train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size=0.5)

reshaped = lambda x: x.reshape(x.shape[0], x.shape[1], 1)


test_X = reshaped(test_X)
train_X = reshaped(train_X)
val_X = reshaped(val_X)

test_Y, test_inverse = snipY(test_Y, invList)
train_Y, train_inverse = snipY(train_Y, invList)
val_Y, val_inverse = snipY(val_Y, invList)


model = models.Sequential()

# input layer (pre-network convolution)
#model.add(layers.Conv1D(32, kernel_size=8, strides=1, input_shape=(None, 1), activation='swish', padding="causal"))
#model.add(layers.AveragePooling1D(2))

# LSTM
model.add(layers.LSTM(48, activation='swish', input_shape=(None, 1), return_sequences=False))

# hidden layers
model.add(layers.Dense(128, activation='swish'))
model.add(layers.Dense(64, activation='linear'))

# output layer
model.add(layers.Dense(output_count, activation='linear'))

model.summary()

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

history = model.fit(train_X, train_Y, epochs=30, batch_size=64, validation_data=(val_X, val_Y))

# stats and stuff

prediction = model.predict(test_X)

# examples of guesses

for i in range(20):
    plot(1000 - output_count, output_count, test_X, test_Y, prediction, i)
    plot(output_count, output_count, test_X, test_Y, prediction, i)
    

plotLoss(history)

cost = (prediction - test_Y) ** 2
avg_cost_per_node = [None] * len(cost[0])
avg_cost_per_row = [None] * len(cost)

for x in range(0, len(cost)):
    avg_cost_per_row[x] = np.average(cost[x, :])

for x in range(0, len(cost[0])):
    avg_cost_per_node[x] = np.average(cost[:, x])

temp = np.linspace(0, len(cost[0]) * 15, len(cost[0]))

# why. why is this how you call it. who made this
m, b = np.polyfit(temp, avg_cost_per_node, 1)

print("Greatest Error:".rjust(18), "{a:.5}".format(a=np.max(cost)).rjust(10))
print("Smallest Error:".rjust(18), "{a:.5}".format(a=np.min(cost)).rjust(10))
print("Average Error:".rjust(18), "{a:.5}".format(a=np.average(cost)).rjust(10))
print("Median:".rjust(18), "{a:.5}".format(a=np.median(cost)).rjust(10))
print("STD:".rjust(18), "{a:.5}".format(a=np.std(cost)).rjust(10))
print("Degradation Rate:".rjust(18), "{a:.5}".format(a=m * 15).rjust(10))

# average cost per output
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.set(title="Average cost Per Output Node", xlabel='Node', ylabel='Average Cost')
ax.plot(range(0, len(cost[0]), 1), avg_cost_per_node)
plt.plot(range(0, len(cost[0]), 1), m * temp + b)
plt.show()

# highest cost graph
index = smallestIndex(avg_cost_per_row)
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.set(title="Smallest Average Cost Graph (Index of {a})".format(a=index), xlabel='Minutes', ylabel='Scaled Value')
ax.plot(range(15000 - len(cost[0]) * 30, 15000, 15),
        np.concatenate((test_X[index, -output_count:].flatten(),
                        test_Y[index].flatten())))
ax.plot(range(15000 - len(cost[0]) * 15, 15000, 15), prediction[index])
ax.axvline(x=(15000 - output_count * 15), color='green', linewidth=2, linestyle='--')
plt.show()

# lowest cost graph
index = biggestIndex(avg_cost_per_row)
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.set(title="Largest Average Cost Graph (Index of {a})".format(a=index), xlabel='Minutes', ylabel='Scaled Value')
ax.plot(range(15000 - len(cost[0]) * 30, 15000, 15),
        np.concatenate((test_X[index, -output_count:].flatten(),
                        test_Y[index].flatten())))
ax.plot(range(15000 - len(cost[0]) * 15, 15000, 15), prediction[index])
ax.axvline(x=(15000 - output_count * 15), color='green', linewidth=2, linestyle='--')
plt.show()


# would it work?? answer: no

moneyIn = 0
profits = []
maxProfits = []
for i in range(len(test_X)):
    inverse = test_inverse[i]
    
    init = inverse(test_X)[i, -1]
    
    pred = inverse(prediction[i])
    actual = inverse(test_Y[i])
    
    ind = pred.argmax()
    predVal = pred[ind]
    actualVal = actual[ind]
    
    if predVal > init:
        profits.append(actualVal - init)
        moneyIn += init
    
    actualMax = actual[actual.argmax()]
    
    if actualMax > init:
        maxProfits.append(actualMax - init)
    
profits = np.array(profits)
maxProfits = np.array(maxProfits)
profit = profits.sum()
meanProfit = profits.mean()
maxProfit = maxProfits.sum()

print()
print('Money In:'.rjust(12), "{:.2f}".format(float(moneyIn)))
print('Profit:'.rjust(12), "{:.2f}".format(profit))
print('Max Profit:'.rjust(12), "{:.2f}".format(maxProfit))








