import logging
import threading
import time
import multiprocessing

from pythonProject import neuralNet
from pythonProject.getData import pastData

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")


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


def Predict():
    while True:
        time.sleep(10)
        logging.info("asdf")


if __name__ == "__main__":

    logging.info("Starting Threads")

    threads = list()

    updateNN = threading.Thread(target=UpdateNN)
    threads.append(updateNN)

    predict = threading.Thread(target=Predict)
    threads.append(predict)

    updateNN.start()
    predict.start()

    logging.info("Threads started")


