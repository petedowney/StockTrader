import logging
import os
import threading
import time
import multiprocessing
import traceback

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

        #print(api.list_positions())
        #print(api.get_barset())

    except trade_api.rest.APIError:
        print("Invalid Keys")
        exit(1)

    assert api is not None, "API is null"

    # logging
    format = "%(asctime)s: %(message)s"
    #logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    '''_listening = np.array(["AAPL",
    "AAON",
    "AAWW",
    "ABMD", "ABNB", "ACAD", "GME", "CACI", "CCMP"])'''

    _listening = past_data.get_symbols();

    # updated data from stocks that are being listened to
    data, remove = past_data.get_past_data(api, _listening)

    _listening = np.setdiff1d(_listening, remove.tolist(), True)

    """assert _listening.shape[0] == data.shape[0], \
        f"some in listening array do not have full 1000 data " \
        f"{_listening.shape[0]} + " f"{data.shape[0]}"
"""
    adding_data_semaphore = threading.Semaphore()

    @staticmethod
    def get_listening():
        return Main._listening


# listens to the alpaca api data stream
def listen_to_data():
    try:
        current_data.listen()
        print("asdf")
    except Exception as exc:
        print(str(type(exc))[8:-2])
        print(exc.args)
        print(traceback.format_tb(exc.__traceback__))
        exit(1)


# updates the NN on past data every 15 min or so
def update_nn():
    x = 0
    while True:

        try:
            x += 1

            # updates NN
            #logging.info("Updating NN")
            neural_net.train_neural_net(Main.data)

            # sleeps for 10 minutes
            #logging.info(str(x) + " Cycles Completed")
            time.sleep(10 * 60)

        except Exception as exc:
            print(str(type(exc))[8:-2])
            print(exc.args)
            print(traceback.format_tb(exc.__traceback__))
            exit(1)


# takes streamed data and creates prediction data
def predict():
    time.sleep(60 * 8)
    try:
        prediction.create_stocks(Main.get_listening())
        while True:
            Main.data = prediction.update_data(Main.data, Main.get_listening())
            #prediction.predict_graph(Main.data)
            prediction.buy_and_sell(Main.data)

            # sleeps for 1 minute
            time.sleep(60)
    except Exception as exc:
        print(str(type(exc))[8:-2])
        print(exc.args)
        print(traceback.format_tb(exc.__traceback__))
        exit(1)

if __name__ == "__main__":
    #logging.info("Starting Threads")

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

    listen_to_data.start()
    update_nn.start()
    predict.start()

    #logging.info("Threads started")
