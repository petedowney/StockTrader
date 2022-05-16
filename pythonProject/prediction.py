import logging
import traceback
import numpy as np
import matplotlib.pyplot as plt
import copy
from pythonProject.getData import current_data
from pythonProject import stock, main, neural_net


class Prediction:
    stocks = []
    money = 100000
    # logging.basicConfig(filename="stocks_to_remove.txt",
    # filemode='a',
    # format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    # datefmt='%H:%M:%S',
    # level=logging.DEBUG)

    # logging.info("Buying and Selling Stocks")


# Updates the data for predictions
def update_data(data, listening):

    x = current_data.get_new_data()

    for key in x.keys():

        n = -1
        for company in range(0, listening.shape[0]):
            if key == listening[company]:
                n = company
                break

        assert n != -1, f"Index not found in listening {key} \n{listening}"

        for time_stamp in x[key]:
            past_ema = data[n, -1, -1]
            new_ema = time_stamp[0] * (2 / 1001) + \
                      past_ema * (1 - (2 / 1001))
            new_data = np.append(time_stamp, [new_ema])

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

    model = neural_net.get_model()

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
    # average = np.expand_dims(np.average(data, axis=1), 1)

    # prediction is in array format
    prediction = get_prediction_data(data)
    # avg_prediction = get_prediction_data(average)

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set(title="Distribution After Scaling", xlabel='Minutes', ylabel='Stock Scaled Value')

    for company_index, company in enumerate(prediction):
        ax.plot(range(15000, 15750, 15), company, '--')
        ax.plot(range(0, 15000, 15), data[company_index, 0])


    plt.show()


# needs to be called before buy_and_sell
# creates the stock classes
def create_stocks(listening):
    Prediction.stocks = [stock.Stock(name) for name in listening]


# in charge of buying and selling data
def buy_and_sell(data):

    # gets the un-normalized prediction data
    prediction = get_unnormalized_prediction_data(data)

    profit = 0

    for company in range(0, data.shape[0]):
        if Prediction.stocks[company].amount == 0 and \
                Prediction.stocks[company].should_buy(data[company, 0, -1], prediction[company], 1.01) and \
                not Prediction.stocks[company].should_sell(data[company, 0, -1], prediction[company], .9):
            try:
                Prediction.stocks[company].buy(main.Main.api, data[company, 0, -1], 1)

            except Exception as exc:

                logging.debug(f"Remove: {main.Main._listening[company]}")
                print(str(type(exc))[8:-2])
                print(exc.args)
                print(traceback.format_tb(exc.__traceback__))
                exit(1)

        elif Prediction.stocks[company].amount != 0 and Prediction.stocks[company].should_sell(data[company, 0, -1],
                                                                                               prediction[company],
                                                                                               .9):
            Prediction.stocks[company].sell(main.Main.api, data[company, 0, -1])

        profit += Prediction.stocks[company].profit

    print(profit)
