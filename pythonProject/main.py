import logging
import threading
import time
import multiprocessing

from pythonProject import neuralNet, prediction
from pythonProject.getData import pastData, currentData


format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

data = pastData.PastData2()

class Main:

    @staticmethod
    def ListenToData():
        currentData.Data.listen()

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

    @staticmethod
    def Predict():

        print("asdf")

        while True:
            time.sleep(10)
            Main.data = prediction.updateData(data)
            prediction.getPredictionData(data)


if __name__ == "__main__":

    logging.info("Starting Threads")

    threads = list()

    listenToData = threading.Thread(target=Main.ListenToData)
    threads.append(listenToData)

    updateNN = threading.Thread(target=Main.UpdateNN)
    threads.append(updateNN)

    predict = threading.Thread(target=Main.Predict)
    threads.append(predict)

    updateNN.start()
    predict.start()

    logging.info("Threads started")


