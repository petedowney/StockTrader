import time

from keras.models import model_from_json
import numpy as np
from sklearn.model_selection import train_test_split

from pythonProject.getData import pastData, currentData

from pythonProject import methods


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
    sData, meanList, rangeList = methods.standerdize(data)


    reshaped = lambda x: x.reshape(x.shape[0], x.shape[1], 1)

    sData = reshaped(sData)

    #code from:
    #https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    json_file = open('data/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("data/model.h5")

    prediction = model.predict(sData)

    return prediction