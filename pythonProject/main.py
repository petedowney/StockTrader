import logging
import os
import threading
import time
import multiprocessing
import numpy as np

import alpaca_trade_api as trade_api
from pythonProject import neural_net, prediction
from pythonProject.getData import past_data, current_data, config


class Main:
    # connecting to the alpaca API and alpaca account
    api = None
    try:
        api = trade_api.REST(config.API_KEY, config.SECRET_KEY,
                             base_url='https://paper-api.alpaca.markets',
                             api_version="v2")
        api.list_positions()

    except trade_api.rest.APIError:
        print("Invalid Keys")
        exit(1)

    assert api is not None, "API is null"

    # logging
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    listening = np.array(["AAPL",
"AAON",
"AAWW",
"ABMD", "ABNB", "ACAD", "GME", "CACI", "CCMP"
])#past_data.get_symbols()

    # updated data from stocks that are being listened to
    data = past_data.get_past_data(api, listening)

    adding_data_semaphore = threading.Semaphore()


# listens to the alpaca api data stream
def listen_to_data():
    current_data.listen()


# updates the NN on past data every 15 min or so
def update_nn():
    x = 0
    while True:
        x += 1

        # retrieves data
        logging.info("Retrieving Data ")
        # TODO make this more efficent by instead getting the data from the running data list
        past_data.get_past_data_to_csv(Main.api)

        # updates NN
        logging.info("Updating NN")
        neural_net.train_neural_net()

        # sleeps for 10 minutes
        logging.info(str(x) + " Cycles Completed")
        time.sleep(10 * 60)


# takes streamed data and creates prediction data
def predict():

    prediction.create_stocks(Main.listening)
    while True:
        Main.data = prediction.update_data(Main.data, Main.listening)
        prediction.buy_and_sell(Main.data)

        # sleeps for 1 minute
        time.sleep(60)


if __name__ == "__main__":
    logging.info("Starting Threads")

    # creating the three theads
    threads = list()

    # listens to the alpaca api data stream
    listen_to_data = threading.Thread(target=listen_to_data)
    # threads.append(listenToData)

    # updates the NN on past data every 15 min or so
    update_nn = threading.Thread(target=update_nn)
    # threads.append(updateNN)

    # takes streamed data and creates prediction data
    predict = threading.Thread(target=predict)
    # threads.append(predict)

    #listen_to_data.start()
    update_nn.start()
    #predict.start()

    logging.info("Threads started")
