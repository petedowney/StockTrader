

from keras.models import model_from_json
import numpy
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from pythonProject import methods
from pythonProject.getData import pastData, config
import alpaca_trade_api as tradeapi




api = None
try:
    api = tradeapi.REST(config.APIKEY, config.SECRETKEY,
                        base_url='https://paper-api.alpaca.markets',
                        api_version="v2")
    api.list_positions()

except tradeapi.rest.APIError:
    print("Invalid Keys")
    exit(1)

assert api is not None


data = pastData.PastData(api, np.array(["GME", "APPL", "GOOGL"]), 10)

#print(data)
