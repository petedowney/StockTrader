import logging
import threading
import time
import multiprocessing
import numpy as np

import alpaca_trade_api as tradeapi
from pythonProject import neuralNet, prediction, buyerAndSeller
from pythonProject.getData import pastData, currentData, config


class Main:

    api = None
    try:
        api = tradeapi.REST(config.APIKEY, config.SECRETKEY,
                            base_url='https://paper-api.alpaca.markets',
                            api_version="v2")
        api.list_positions()

    except tradeapi.rest.APIError:
        print("Invalid Keys")
        exit(1)

    assert api is not None

    # logging
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    listening = pastData.getSymbols()

    data = pastData.PastData2(api, listening)

    addingDataSemaphore = threading.Semaphore()


    # updates stocks
    @staticmethod
    def BuyAndSell():
        buyerAndSeller.BuyerAndSeller.updateStockPositions()

    # listens to the alpaca api data stream
    @staticmethod
    def ListenToData():
        currentData.Data.listen()

    # updates the NN on past data every 15 min or so
    @staticmethod
    def UpdateNN():

        x = 0
        while True:
            x += 1
            logging.info("Retrieving Data ")
            pastData.PastData(Main.api)

            logging.info("Updating NN")
            neuralNet.NeuralNet()

            logging.info(str(x) + " Cycles Completed")
            time.sleep(10 * 60)

    # takes streamed data and creates prediction data
    @staticmethod
    def Predict():

        while (True):
            Main.data = prediction.updateData(Main.data)
            prediction.predictGraph(Main.data)
            prediction.predict(Main.data)
            time.sleep(60)

if __name__ == "__main__":
    logging.info("Starting Threads")

    # creating the three theads
    threads = list()

    # listens to the alpaca api data stream
    listenToData = threading.Thread(target=Main.ListenToData)
    threads.append(listenToData)

    # updates the NN on past data every 15 min or so
    updateNN = threading.Thread(target=Main.UpdateNN)
    threads.append(updateNN)

    # takes streamed data and creates prediction data
    predict = threading.Thread(target=Main.Predict)
    threads.append(predict)

    listenToData.start()
    updateNN.start()
    predict.start()

    logging.info("Threads started")
