import numpy as np


class Stock:

    name = ""
    bought_price = 0
    amount = 0
    profit = 0

    def __init__(self, name):
        self.name = name

    def buy(self, api, bought_price, amount):
        self.bought_price = bought_price
        self.amount = amount
        api.submit_order(self.name, amount)

    def sell(self, api, curr_price):
        self.profit += (curr_price - self.bought_price) * self.amount
        api.submit_order(self.name, self.amount, side="sell")
        self.amount = 0

    def should_sell(self, curr_price, prediction, acc_risk):
        return self.amount != 0 and curr_price / self.bought_price < acc_risk or self.expected_slope(prediction)

    def should_buy(self, curr_price, prediction, buy_threshold):
        return max(prediction) / curr_price > buy_threshold and self.expected_slope(prediction)

    def expected_slope(self, prediction):
        return np.argmax(prediction) < np.argmin(prediction) and prediction[0] < max(prediction)