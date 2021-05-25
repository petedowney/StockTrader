from keras.models import model_from_json
import numpy
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from pythonProject import methods

# DATA ===========
file_name = 'data/techData.csv'
raw_data = open(file_name, 'rt')
data = np.loadtxt(raw_data, delimiter=',', dtype=np.float)

output_count = 50

# standardization
data, meanList, rangeList = methods.standerdize(data)


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


#code from:
#https://machinelearningmastery.com/save-load-keras-deep-learning-models/
json_file = open('data/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("data/model.h5")
print("Loaded model from disk")

prediction = model.predict(test_X)

cost = (prediction - test_Y) ** 2
avg_cost_per_node = [None] * len(cost[0])
avg_cost_per_row = [None] * len(cost)

for x in range(0, len(cost)):
    avg_cost_per_row[x] = np.average(cost[x, :])

for x in range(0, len(cost[0])):
    avg_cost_per_node[x] = np.average(cost[:, x])

temp = np.linspace(0, len(cost[0]) * 15, len(cost[0]))

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
index = methods.smallestIndex(avg_cost_per_row)
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
index = methods.biggestIndex(avg_cost_per_row)
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.set(title="Largest Average Cost Graph (Index of {a})".format(a=index), xlabel='Minutes', ylabel='Scaled Value')
ax.plot(range(15000 - len(cost[0]) * 30, 15000, 15),
        np.concatenate((test_X[index, -output_count:].flatten(),
                        test_Y[index].flatten())))
ax.plot(range(15000 - len(cost[0]) * 15, 15000, 15), prediction[index])
ax.axvline(x=(15000 - output_count * 15), color='green', linewidth=2, linestyle='--')
plt.show()

# profit predictions

moneyIn = 0
profits = []
maxProfits = []
for i in range(len(test_X)):
    invIndex = test_inverse[i]
    inverse = lambda x: x * rangeList[invIndex] + meanList[invIndex]

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
