

from keras.models import model_from_json
import numpy
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from pythonProject import methods


hello = [[y for y in range(100)] for x in range(100)]

hello = np.array(hello)

print(hello)

X, Y = methods.splitData(hello, 21)

reshaped = lambda f: f.reshape(f.shape[0], f.shape[1], 1)

X = np.array(reshaped(X))
Y = np.array(reshaped(Y))

#print(X)
print(Y)
