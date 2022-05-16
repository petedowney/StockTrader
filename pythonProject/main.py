import logging
import threading
import time
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

    except trade_api.rest.APIError:
        print("Invalid Keys")
        exit(1)

    assert api is not None, "API is null"

    # logging
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    # gets all stocks that are being listened to
    _listening = past_data.get_symbols();

    # updated data from stocks that are being listened to
    data, remove = past_data.get_past_data(api, _listening)

    # removes any listening that dont meet training reqs
    _listening = np.setdiff1d(_listening, remove.tolist(), True)

    assert _listening.shape[0] == data.shape[0], \
        f"some in listening array do not have full 1000 data " \
        f"{_listening.shape[0]} + " f"{data.shape[0]}"

    adding_data_semaphore = threading.Semaphore()

    # returns listening companies
    @staticmethod
    def get_listening():
        return Main._listening


# listens to the alpaca api data stream
def listen_to_data():
    try:
        logging.info("Listening to new data")
        current_data.listen()

    except Exception as exc:
        print(str(type(exc))[8:-2])
        print(exc.args)
        print(traceback.format_tb(exc.__traceback__))
        exit(1)


# updates the NN on past data every 15 min or so
def update_nn():
    n = 0
    while True:

        try:
            n += 1

            # updates NN
            logging.info("Updating NN")
            neural_net.train_neural_net(Main.data)

            # sleeps for 10 minutes
            logging.info(str(n) + " Training Cycles Completed")
            time.sleep(10 * 60)

        except Exception as exc:
            print(str(type(exc))[8:-2])
            print(exc.args)
            print(traceback.format_tb(exc.__traceback__))
            exit(1)


# takes streamed data and creates prediction data
def predict():
    time.sleep(60 * 15)
    n = 0
    try:
        prediction.create_stocks(Main.get_listening())
        while True:
            n += 1

            Main.data = prediction.update_data(Main.data, Main.get_listening())

            # this line will create prediction graphs
            # prediction.predict_graph(Main.data)

            # buys and sells the stock
            prediction.buy_and_sell(Main.data)

            logging.info(str(n) + " Prediction Cycles Completed")

            # sleeps for 1 minute
            time.sleep(60)

    except Exception as exc:
        print(str(type(exc))[8:-2])
        print(exc.args)
        print(traceback.format_tb(exc.__traceback__))
        exit(1)


if __name__ == "__main__":
    logging.info("Starting Threads")

    # listens to the alpaca api data stream
    listen_to_data = threading.Thread(target=listen_to_data)

    # updates the NN on past data every 15 min or so
    update_nn = threading.Thread(target=update_nn)

    # takes streamed data and creates prediction data
    predict = threading.Thread(target=predict)

    listen_to_data.start()
    update_nn.start()
    predict.start()

    logging.info("Threads started")
