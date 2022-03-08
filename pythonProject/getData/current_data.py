from pythonProject.getData import config
from pythonProject import main

import websocket, json
import numpy as np


class Data:
    # Determines which companies the program will listen to
    out_put_message = {}


# When the listening thread is opened will validate the connection
# and begin listening to the listenCompanies
def on_open(ws):
    print("open")
    auth_data = {
        "action": "authenticate",
        "data": {"key_id": config.API_KEY, "secret_key": config.SECRET_KEY}
    }

    ws.send(json.dumps(auth_data))

    listen_message = {"action": "listen",
                      "data": {"streams": ["AM." + x for x in main.Main.listening]}}
    ws.send(json.dumps(listen_message))


# On a message will update the outPutMessage with the new data
# If it is an authentication message it will just print it out
def on_message(ws, message):
    message = json.loads(message)

    if "o" in message.get("data"):

        main.Main.adding_data_semaphore.acquire()

        if message["stream"] in Data.out_put_message.keys():
            Data.out_put_message[message["stream"]] = np.append(Data.out_put_message[message["stream"]],
                                                                [message.get("data").get("o"),
                                                                 message.get("data").get("v"),
                                                                 message.get("data").get("h") -
                                                                 message.get("data").get("l")], axis=1)
        else:
            Data.out_put_message[message["stream"]] = \
                [message.get("data").get("o"),
                 message.get("data").get("v"),
                 message.get("data").get("h") -
                 message.get("data").get("l")]

        main.Main.adding_data_semaphore.release()
    else:
        print(message)


# just logs that the connection has been closed
def on_close(ws):
    print("closed connection")


# will listen to companies for new data based on the standerdListen
def listen():
    socket = "wss://data.alpaca.markets/stream"
    ws = websocket.WebSocketApp(socket, on_open=on_open, on_message=on_message, on_close=on_close)
    ws.run_forever()


# empties the outPutMessage and adds it to the main data
def get_new_data():
    main.Main.adding_data_semaphore.acquire()

    temp = dict.copy(Data.out_put_message)
    Data.out_put_message.clear()

    main.Main.adding_data_semaphore.release()
    return temp;
