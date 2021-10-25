import logging
import threading
import time
import multiprocessing
import numpy as np

from pythonProject import neuralNet, prediction
from pythonProject.getData import pastData, currentData


class Main:

    # logging
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    data = pastData.PastData2(np.array(["AAPL"]))

    addingDataSemaphore = threading.Semaphore()

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
            pastData.PastData()

            logging.info("Updating NN")
            neuralNet.NeuralNet()

            logging.info(str(x) + " Cycles Completed")
            time.sleep(10 * 60)

    # takes streamed data and creates prediction data
    @staticmethod
    def Predict():

        while (True):
            Main.data = prediction.updateData(Main.data)
            time.sleep(20)


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
