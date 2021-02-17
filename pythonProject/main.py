
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

api = tradeapi.REST('<key_id>', '<secret_key>', base_url='https://paper-api.alpaca.markets')
account = api.get_account()
api.list_positions()