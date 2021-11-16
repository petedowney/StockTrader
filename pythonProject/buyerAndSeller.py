import json

from pythonProject.getData import config

import alpaca_trade_api as tradeapi

class BuyerAndSeller:

    account = None

    @staticmethod
    def updateStockPositions(ws, data):

        auth_data = {
            "action": "authenticate",
            "data": {"key_id": config.APIKEY, "secret_key": config.SECRETKEY}
        }

        ws.send(json.dumps(auth_data))

        #acount =

        for keys in data.keys():
            f =0
