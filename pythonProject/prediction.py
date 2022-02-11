import time

from keras.models import model_from_json
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import copy
from pythonProject.getData import past_data, current_data

from pythonProject import methods


# standerdizes data
def standerdize_data(data):

    data2 = copy.deepcopy(data)

    for company in range(0, data.shape[0]):
        for inputType in range(0, data.shape[1]):
            mean = data[company, inputType].mean()
            ranges = data[company, inputType].max() - data[company, inputType].min()

            data2[company, inputType] = (data[company, inputType] - mean) / ranges

    return data2

# Updates the data for predictions
def update_data(data, listening):

    x = current_data.Data.get_new_data()

    for key in x.keys():
        n = np.where(listening, key)

        data[n]

        data[n] = np.append(data[n], np.array(x[key]).reshape((3,1)), axis=1)
        data[n] = data[n, :, -1000:]

    return data

# With the updated data create new predictions
def get_prediction_data(p_data):
    #a1 = data[:,:,:949]
    #a2 = data[:,:,-949:]

    # reformating data from
    # (channels, company, 1000) -> (company, 1, channels, 950)

    data = np.expand_dims(np.swapaxes(p_data[:, :, -950:], 0, 1), axis=1)

    # code from:
    # https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    json_file = open('data/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("data/model.h5")

    # returns prediction and converts to a 2d array of (company, time index)
    prediction = np.squeeze(model.predict(data))

    return prediction

def predict_graph(data1):

    #standerdizes data and finds average
    data = standerdize_data(data1)
    average = np.expand_dims(np.average(data, axis=1), 1)

    # prediction is in array format
    prediction = get_prediction_data(data)
    avg_prediction = get_prediction_data(average)
    #n = 0
    #for key in data.keys():
    #    data[key] = np.append(np.squeeze(data[key][0, :, -950:]), prediction[n])
    #    n+=1

    #prediction2 = get_prediction_data(np.append(np.squeeze(data[0, -900:]), prediction))

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set(title= "Distribution After Scaling", xlabel='Minutes', ylabel='Stock Scaled Value')

    for company_index, company in enumerate(prediction):
        ax.plot(range(15000, 15750, 15), company, '--')
        ax.plot(range(0, 15000, 15), data[0, company_index])

    ax.plot(range(15000, 15750, 15), avg_prediction, '.-')
    ax.plot(range(0, 15000, 15), average[0, 0], '.-')

    #for n in prediction2:
    #    ax.plot(range(15750, 16500, 15), n)

    plt.show()

def predict(data1):

    # data is in dict format
    data = standerdize_data(data1)

    # prediction is in array format
    prediction = get_prediction_data(data)

    # TODO work on this algorthim
    for key in data.keys():

        minIndex = 0
        minValue = 100;
        maxIndex = 0
        maxValue = -100;
        initialValue = data.get(key)[-1:]

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



