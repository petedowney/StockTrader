import config
import websocket, json

listening_to = ["AM.TSLA", "AM.GME"]


def on_open(ws):
    print("open")
    auth_data = {
        "action": "authenticate",
        "data": {"key_id": config.API_KEY, "secret_key": config.SECRET_KEY}
    }

    ws.send(json.dumps(auth_data))

    listen_message = {"action": "listen", "data": {"streams": listening_to}}
    ws.send(json.dumps(listen_message))


def on_message(ws, message):
    print("received a message")
    print(message)


def on_close(ws):
    print("closed connection")

socket = "wss://data.alpaca.markets/stream"

ws = websocket.WebSocketApp(socket, on_open=on_open, on_message=on_message, on_close=on_close)
ws.run_forever()