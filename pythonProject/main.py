import logging
import threading
import time

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
        time.sleep(60)

if __name__ == "__main__":

    logging.info("Starting Threads")

    updateNN = threading.Thread(target=UpdateNN(), args=(1,))
    updateNN.start()

    predict = threading.Thread(target=Predict(), args=(1,))
    predict.start()

    logging.info("Threads started")


