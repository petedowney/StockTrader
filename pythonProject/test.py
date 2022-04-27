from datetime import date
from datetime import timedelta
from datetime import datetime

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

today = datetime.utcnow() - timedelta(minutes=15)
#today = today.se
last_week = today - timedelta(days=2)

today = today.isoformat("T") + "Z"
last_week = last_week.isoformat("T") + "Z"


start_date = datetime.utcnow().replace(hour=9, minute=30) - timedelta(days=1)#(datetime.utcnow() - timedelta(days=2)).isoformat("T") + "Z"
end_date = datetime.utcnow().replace(hour=14, minute=00) - timedelta(days=1)

calander = api.get_calendar((start_date+timedelta(days=1)).isoformat("T") + "Z", (end_date+timedelta(days=1)).isoformat("T") + "Z")

#api.get_calendar((start_date-timedelta(10)).isoformat("T") + "Z", (start_date-timedelta(10)).isoformat("T") + "Z")[0]._raw["date"]

minute_bars = api.get_bars("AAPL", "1Min", end=str(today), start=last_week, limit=2000).df