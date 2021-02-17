from keras.datasets import mnist
from keras import models
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow
from keras.utils import to_categorical
from keras.models import load_model
import h5py
import alpaca_trade_api as tradeapi
import json


try:
    print("Accessing Keys")

    with open("keys.json", "r") as read_file:
        data = json.load(read_file)

    print("Keys Found")
    print("Accessing Account")

    api = tradeapi.REST(data["Public Key"], data["Private Key"], base_url='https://paper-api.alpaca.markets')
    account = api.get_account()
    api.list_positions()

    print("Account Accessed")

except FileNotFoundError:
    print("Keys not found")
    exit(1)

except tradeapi.rest.APIError:
    print("Invalid Keys")
    exit(2)

