from pythonProject.getData import config
from pythonProject import main

import websocket, json
import numpy as np


class Data:

    # Determines which companies the program will listen to
    outPutMessage = {}

    # When the listening thread is opened will validate the connection
    # and begin listening to the listenCompanies
    @staticmethod
    def onOpen(ws):

        print("open")
        auth_data = {
            "action": "authenticate",
            "data": {"key_id": config.APIKEY, "secret_key": config.SECRETKEY}
        }

        ws.send(json.dumps(auth_data))

        listen_message = {"action": "listen",
                          "data": {"streams": ["AM." + x for x in main.Main.listening]}}
        ws.send(json.dumps(listen_message))

    # On a message will update the outPutMessage with the new data
    # If it is an authentication message it will just print it out
    @staticmethod
    def onMessage(ws, message):
        message = json.loads(message)

        if "o" in message.get("data"):

            main.Main.addingDataSemaphore.acquire()

            if message["stream"] in Data.outPutMessage.keys():
                Data.outPutMessage[message["stream"]] = np.append(Data.outPutMessage[message["stream"]],
                                                                  message.get("data").get("o"))
            else:
                Data.outPutMessage[message["stream"]] = [message.get("data").get("o")]

            main.Main.addingDataSemaphore.release()
        else:
            print(message)

    # just logs that the connection has been closed
    @staticmethod
    def onClose(ws):
        print("closed connection")

    # will listen to companies for new data based on the standerdListen
    @staticmethod
    def listen():

        socket = "wss://data.alpaca.markets/stream"
        ws = websocket.WebSocketApp(socket, on_open=Data.onOpen, on_message=Data.onMessage, on_close=Data.onClose)
        ws.run_forever()

    # empties the outPutMessage and adds it to the main data
    @staticmethod
    def getNewData():

        main.Main.addingDataSemaphore.acquire()

        temp = dict.copy(Data.outPutMessage)
        Data.outPutMessage.clear()

        main.Main.addingDataSemaphore.release()
        return temp;


if __name__ == "__main__":
    Data.listen()
