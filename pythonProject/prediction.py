import time

from keras.models import model_from_json
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

from pythonProject.getData import past_data, current_data

from pythonProject import methods


# standerdizes data
def standerdize_data(data):

    data2 = dict.copy(data)

    for key in data.keys():

        for inputType in range(0, data[key].shape[0]):
            mean = data[key][inputType].mean()
            ranges = data[key][inputType].max() - data[key][inputType].min()

            data2[key][inputType] = (data[key][inputType] - mean) / ranges

    return data2

# Updates the data for predictions
def update_data(data):

    x = current_data.Data.getNewData()

    for keys in x.keys():
        data[keys] = np.append(data.get(keys), x.get(keys))
        data[keys] = data[keys][-1000:]

    return data

# With the updated data create new predictions
def get_prediction_data(data):

    # converts data into an array
    data = np.swapaxes(np.array(list(data.values())), 0, 1)

    #a1 = data[:,:,:949]
    #a2 = data[:,:,-949:]


    # I am so confused
    # when doing data[:,:,:949] it predicts the next 50 really well
    # however if its data[:,:,-949:] it makes no sense

    # reformating data from
    # (2, 3, 1000) -> (?, 1, 2, 949)
    sArrayData = np.expand_dims(np.swapaxes(data[:,:,-949:], 0, 1), axis=1)

    # code from:
    # https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    json_file = open('data/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("data/model.h5")

    # returns prediction and converts to a 2d array of (company, time index)
    prediction = np.squeeze(model.predict(sArrayData))

    return prediction

def predict_graph(data):

    # data is in dict format
    data = standerdize_data(data)

    # prediction is in array format
    prediction = get_prediction_data(data)

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set(title= "Distribution After Scaling", xlabel='Minutes', ylabel='Stock Scaled Value')

    for n in prediction:
        ax.plot(range(15000, 15735, 15), n[:-1])

    for key in data.keys():
        ax.plot(range(0, 15000, 15), data[key][0])

    plt.show()

def predict(data):

    # data is in dict format
    data = standerdize_data(data)

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



