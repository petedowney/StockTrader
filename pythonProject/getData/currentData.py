
from pythonProject.getData import config
import websocket, json
import numpy as np

standardListen = ["AM.AAPL"]

listen = []
outPutMessage = np.array([]);

class Data:

    @staticmethod
    def on_open(ws):

        print("open")
        auth_data = {
            "action": "authenticate",
            "data": {"key_id": config.API_KEY, "secret_key": config.SECRET_KEY}
        }

        ws.send(json.dumps(auth_data))

        listen_message = {"action": "listen", "data": {"streams": listen}}
        ws.send(json.dumps(listen_message))

    @staticmethod
    def on_message(ws, message):
        print("received a message")
        print(message)

    @staticmethod
    def on_message2(ws, message):
        print("received a message")
        print(message)
        np.append(outPutMessage, message);

    @staticmethod
    def on_close(ws):
        print("closed connection")

    @staticmethod
    def listen(listenTo = standardListen):

        Data.listen = listenTo
        socket = "wss://data.alpaca.markets/stream"
        ws = websocket.WebSocketApp(socket, on_open=Data.on_open, on_message=Data.on_message2, on_close=Data.on_close)
        ws.run_forever()

    @staticmethod
    def getNewData():
        temp = outPutMessage
        Data.outPutMessage = np.array([]);
        return temp;

if __name__ == "__main__":
    Data.listen()