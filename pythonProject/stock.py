import numpy as np


# creates a stock class to handle a stock company
class Stock:
    name = ""
    bought_price = 0
    amount = 0
    profit = 0

    # name: name of stock
    def __init__(self, name):
        self.name = name

    # buys a stock
    def buy(self, api, bought_price, amount):
        api.submit_order(self.name, amount)
        self.bought_price = bought_price
        self.amount = amount

    # sells all of a stock
    def sell(self, api, curr_price):
        self.profit += (curr_price - self.bought_price) * self.amount
        api.submit_order(self.name, self.amount, side="sell")
        self.amount = 0

    # detects if a stock should sell
    def should_sell(self, curr_price, prediction, acc_risk):
        return self.amount != 0 and (curr_price / self.bought_price < acc_risk or not self.expected_slope(prediction))

    # detects if a stock should buy
    def should_buy(self, curr_price, prediction, buy_threshold):
        return max(prediction) / curr_price > buy_threshold and \
               self.expected_slope(prediction) and \
               curr_price < prediction[0]

    # finds the expected slope of a stock
    # true = positive slope | false = negative slope
    @staticmethod
    def expected_slope(self, prediction):
        return np.argmax(prediction) < np.argmin(prediction) and np.argmax(prediction) != 0
