import time

from keras.models import model_from_json
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

from pythonProject.getData import pastData, currentData

from pythonProject import methods


def standerdizeData(data):
    meanList = []
    rangeList = []

    data2 = dict.copy(data)

    for key in data.keys():
        mean = data[key].mean()
        ranges = data[key].max() - data[key].min()

        data2[key] = (data[key] - mean) / ranges

        meanList.append(mean)
        rangeList.append(ranges)

    return data2, meanList, rangeList

# Updates the data for predictions
def updateData(data):

    x = currentData.Data.getNewData()

    for keys in x.keys():
        data[keys] = np.append(data.get(keys), x.get(keys))
        data[keys] = data[keys][-1000:]

    return data

# With the updated data create new predictions
def getPredictionData(data):

    # standardization
    sData, meanList, rangeList = standerdizeData(data)

    sArrayData = np.array(())

    for key in sData.keys():
        sArrayData = np.append(sArrayData, sData[key])

    sArrayData = sArrayData.reshape((len(sData.keys()), 1000, 1))



    # code from:
    # https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    json_file = open('data/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("data/model.h5")

    prediction = model.predict(sArrayData)

    return prediction

def predictGraph(data):

    prediction = getPredictionData(data)

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set(title= "Distribution After Scaling", xlabel='Minutes', ylabel='Stock Scaled Value')

    sData, a, s2 = standerdizeData(data)
    del(a)
    del(s2)

    for n in prediction:
        ax.plot(range(15000, 15735, 15), n[:-1])

    for key in sData.keys():
        ax.plot(range(0, 15000, 15), sData[key])

    plt.show()

def predict(data):
    prediction = getPredictionData(data)

    sData, a, a2 = standerdizeData(data)
    del (a)
    del (a2)
    newPrediction = {}

    n = 0
    for key in data.keys():
        newPrediction[key] = prediction[n]
        n += 1;

    prediction = dict.copy(newPrediction)
    del(newPrediction)

    for key in prediction.keys():

        minIndex = 0
        minValue = 100;
        maxIndex = 0
        maxValue = -100;
        initialValue = sData.get(key)[-1:]

        for x in range(0, len(prediction[key])):
            if prediction.get(key)[x] < minValue:
                minValue = prediction.get(key)[x]
                minIndex = x

            if prediction.get(key)[x] > maxValue:
                maxValue = prediction.get(key)[x]
                maxIndex = x

        print(key)
        print(str(minValue) + " " + str(maxValue))
        print(str(minIndex) + " " + str(maxIndex))
        print(initialValue)

        if minIndex < maxIndex and minValue >= initialValue:
            print("buy")
        else:
            print("sell")



