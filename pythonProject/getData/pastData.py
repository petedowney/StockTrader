import alpaca_trade_api as tradeapi
import datetime
import numpy as np

from pythonProject.getData import config
# requires config file with alpaca-trade-api credentials

def connectToAccount():
    try:

        api = tradeapi.REST(config.API_KEY, config.SECRET_KEY, base_url='https://paper-api.alpaca.markets')
        api.list_positions()

        #print("Account Accessed")

        return api, api.get_account()

    except tradeapi.rest.APIError:
        print("Invalid Keys")
        exit(1)

api, account = connectToAccount()



def vectorize(bars):
    arr = np.zeros(len(bars))
    for i, bar in enumerate(bars):
        arr[i] = bar._raw['o'];
    return arr
    
def getSymbols():

    #the source directory used is the one used from the program that runs it
    fileName = 'data/tech.csv'
    raw_data = open(fileName, 'rt')
    data = np.loadtxt(raw_data, usecols = (0), skiprows = 1, delimiter = ',', dtype = np.str)
    
    return data



def PastData():
    dataLimit = 1000  # amount of samples

    allSymbols = getSymbols()
    data = None

    count = len(allSymbols) // 100 + 1
    for i in range(count):

        e = (i + 1) * 100
        symbols = allSymbols[i * 100: e].tolist()

        barset = api.get_barset(symbols, "15Min", limit=dataLimit)

        vectors = []

        for symbol in symbols:
            vector = vectorize(barset[symbol])
            if (len(vector) == dataLimit):
                vectors.append(vector)

        arr = np.row_stack(tuple(vectors))

        data = arr if (i == 0) else np.row_stack((data, arr))

        #print('progress:', str(i) + "/" + str(count - 1))

    np.savetxt("data/techData.csv", data, delimiter=',', fmt='%f')


if __name__ == "__main__":
    PastData()

