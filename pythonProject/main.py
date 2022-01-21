import logging
import os
import threading
import time
import multiprocessing
import numpy as np

import alpaca_trade_api as trade_api
from pythonProject import neural_net, prediction, buyer_and_seller
from pythonProject.getData import past_data, current_data, config


class Main:

    # connecting to the alpaca API and alpaca account
    api = None
    try:
        api = trade_api.REST(config.APIKEY, config.SECRETKEY,
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

    listening = past_data.getSymbols()

    # updated data from stocks that are being listened to
    data = past_data.PastData2(api, listening)

    addingDataSemaphore = threading.Semaphore()

    # listens to the alpaca api data stream
    @staticmethod
    def listen_to_data():
        current_data.Data.listen()

    # updates the NN on past data every 15 min or so
    @staticmethod
    def update_nn():

        x = 0
        while True:
            x += 1

            # retrieves data
            logging.info("Retrieving Data ")
            # TODO make this more efficent by instead getting the data from the running data list
            past_data.PastData(Main.api)

            # updates NN
            logging.info("Updating NN")
            neural_net.NeuralNet()

            # sleeps for 10 minutes
            logging.info(str(x) + " Cycles Completed")
            time.sleep(10 * 60)

    # takes streamed data and creates prediction data
    @staticmethod
    def predict():

        while (True):
            Main.data = prediction.update_data(Main.data)
            prediction.predict_graph(Main.data)

            # sleeps for 1 minute
            time.sleep(60)


if __name__ == "__main__":

    logging.info("Starting Threads")

    # creating the three theads
    threads = list()

    # listens to the alpaca api data stream
    listenToData = threading.Thread(target=Main.listen_to_data)
    # threads.append(listenToData)

    # updates the NN on past data every 15 min or so
    updateNN = threading.Thread(target=Main.update_nn)
    # threads.append(updateNN)

    # takes streamed data and creates prediction data
    predict = threading.Thread(target=Main.predict)
     #threads.append(predict)

    listenToData.start()
    #updateNN.start()
    predict.start()

    logging.info("Threads started")


