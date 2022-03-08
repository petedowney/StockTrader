import numpy as np
import matplotlib.pyplot as plt
import copy
import random
from pythonProject.getData import current_data
from pythonProject import stock, main
from keras.models import model_from_json


class Prediction:
    stocks = []
    money = 100000


# Updates the data for predictions
def update_data(data, listening):
    x = current_data.get_new_data()

    #for key in listening:
    #    if (random.random() < .2):
    #        x["AM." + key] = [random.random(), random.random(), random.random()]


    assert listening.shape[0] == data.shape[0], "some listening to not have full 1000 data"

    for key in x.keys():

        for company in range(0, listening.shape[0]):
            if key[3:] == listening[company]:
                n = company
                break

        past_ema = data[n, -1, -1]
        new_ema = x[key][0] * (2 / 1001) + \
                past_ema * (1 - (2 / 1001))
        new_data = np.append(np.array(x[key]), new_ema)

        data[n] = np.roll(data[n], -1, axis=1)
        data[n, :, -1] = new_data

    return data


# standerdizes data
def standerdize_data(data):
    data2 = copy.deepcopy(data)

    for company in range(0, data.shape[0]):
        for inputType in range(0, data.shape[1]):
            mean = data[company, inputType].mean()
            ranges = data[company, inputType].max() - data[company, inputType].min()

            data2[company, inputType] = (data[company, inputType] - mean) / ranges

    return data2

# With the updated data create new predictions
# needs normalized data as input
def get_prediction_data(data):

    # reformating data from
    # (company, channels, 1000) -> (company, 1, channels, 950)
    data = np.expand_dims(data[:, :, -950:], axis=1)

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

# gets unnormalized prediction data
# needs unscaled data as input
def get_unnormalized_prediction_data(data):

    s_data = standerdize_data(data)
    prediction = get_prediction_data(s_data)

    for company in range(0, data.shape[0]):
        mean = data[company, 0].mean()
        ranges = data[company, 0].max() - data[company, 0].min()
        prediction[company] = (prediction[company] * ranges) + mean

    return prediction



# creates a graph with past and predicted data for each listening company
def predict_graph(data1):
    # standerdizes data and finds average
    data = standerdize_data(data1)
    average = np.expand_dims(np.average(data, axis=1), 1)

    # prediction is in array format
    prediction = get_prediction_data(data)
    avg_prediction = get_prediction_data(average)

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set(title="Distribution After Scaling", xlabel='Minutes', ylabel='Stock Scaled Value')

    for company_index, company in enumerate(prediction):
        ax.plot(range(15000, 15750, 15), company, '--')
        ax.plot(range(0, 15000, 15), data[0, company_index])

    ax.plot(range(15000, 15750, 15), avg_prediction, '.-')
    ax.plot(range(0, 15000, 15), average[0, 0], '.-')

    plt.show()


# needs to be called before buy_and_sell
# creates the stock classes
def create_stocks(listening):
    Prediction.stocks = [stock.Stock(name) for name in listening]


# in charge of buying and selling data
def buy_and_sell(data):

    # gets the unormalized prediction data
    prediction = get_unnormalized_prediction_data(data)

    profit = 0

    for company in range(0, data.shape[0]):
        if Prediction.stocks[company].amount == 0 and Prediction.stocks[company].should_buy(data[company, 0, -1], prediction[company], 1.01) and not Prediction.stocks[company].should_sell(data[company, 0, -1], prediction[company], 1.1):
            Prediction.stocks[company].buy(main.Main.api, data[company, 0, -1], 1)

        elif Prediction.stocks[company].amount != 0 and Prediction.stocks[company].should_sell(data[company, 0, -1], prediction[company], 1.1):
            Prediction.stocks[company].sell(main.Main.api, data[company, 0, -1])

        profit += Prediction.stocks[company].profit

    print(profit)
