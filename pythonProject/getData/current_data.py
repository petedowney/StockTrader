import traceback

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
    auth_data = {"action":"auth","key":f"{config.API_KEY}",
                 "secret":f"{config.SECRET_KEY}"}

    ws.send(json.dumps(auth_data))

    temp = {"action":"subscribe","bars":["AAPL"]}
    ws.send(json.dumps(temp))

    listen_message = {"action":"subscribe",
                      "bars": main.Main.get_listening()}
    #ws.send(json.dumps(listen_message))


# On a message will update the outPutMessage with the new data
# If it is an authentication message it will just print it out
def on_message(ws, message):
    #print("-------")
    #print(message)
    #print("-------")
    try:
        message = json.loads(message)

        for bar in message:
            if "o" in bar:

                main.Main.adding_data_semaphore.acquire()

                if bar["S"] in Data.out_put_message.keys():

                    Data.out_put_message[bar["S"]] = np.append(Data.out_put_message[bar["S"]],
                                                                        [[bar.get("o"),
                                                                         bar["v"],
                                                                         bar["h"] -
                                                                         bar["l"]]], axis=0)
                else:
                    Data.out_put_message[bar["S"]] = [[bar.get("o"),
                                            bar["v"], bar["h"] - bar["l"]]]

                main.Main.adding_data_semaphore.release()
            else:
                pass
                #print(message)
    except Exception as exc:
        print(str(type(exc))[8:-2])
        print(exc.args)
        print(traceback.format_tb(exc.__traceback__))
        exit(1)



# just logs that the connection has been closed
def on_close(ws):
    print("closed connection")


# will listen to companies for new data based on the standerdListen
def listen():
    socket = "wss://stream.data.alpaca.markets/v2/iex"
    ws = websocket.WebSocketApp(socket, on_open=on_open, on_message=on_message, on_close=on_close)
    ws.run_forever()


# empties the outPutMessage and adds it to the main data
def get_new_data():
    main.Main.adding_data_semaphore.acquire()

    temp = dict.copy(Data.out_put_message)
    Data.out_put_message.clear()

    main.Main.adding_data_semaphore.release()
    return temp;
