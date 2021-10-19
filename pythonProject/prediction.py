import time

from keras.models import model_from_json
import numpy as np
from sklearn.model_selection import train_test_split

from pythonProject.getData import pastData, currentData

from pythonProject import methods

def updateData(data):

    np.append(data, currentData.Data.getNewData())
    data = data[:, -1000:]

def getPredictionData(data):

    # standardization
    data, meanList, rangeList = methods.standerdize(data)


    reshaped = lambda x: x.reshape(x.shape[0], x.shape[1], 1)

    data = reshaped(data)

    #code from:
    #https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    json_file = open('data/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("data/model.h5")

    prediction = model.predict(data)

    print(data[0])
    print(prediction[0])