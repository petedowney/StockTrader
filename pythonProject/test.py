

from keras.models import model_from_json
import numpy
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from pythonProject import methods
from pythonProject.getData import past_data, config
import alpaca_trade_api as tradeapi




api = None
try:
    api = tradeapi.REST(config.API_KEY, config.SECRET_KEY,
                        base_url='https://paper-api.alpaca.markets',
                        api_version="v2")
    api.list_positions()

except tradeapi.rest.APIError:
    print("Invalid Keys")
    exit(1)

assert api is not None


#data = past_data.get_past_data(api, np.array(["GME", "AAPL", "TSLA"]), 10)

#api.submit_order("GME", 1)

api.submit_order("GME", 1, side="sell")
#print(data)

x = 1
#print(data)
print("ad;f;adfsfvzdf;asdf;adf")