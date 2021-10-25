from pythonProject.getData import config

import alpaca_trade_api as tradeapi
import numpy as np

# requires config file with alpaca-trade-api credentials
# connects to the alpaca api account
def connectToAccount():
    try:

        api = tradeapi.REST(config.APIKEY, config.SECRETKEY, base_url='https://paper-api.alpaca.markets')
        api.list_positions()

        #print("Account Accessed")

        return api, api.get_account()

    except tradeapi.rest.APIError:
        print("Invalid Keys")
        exit(1)


api, account = connectToAccount()

# ASKTOBY
def vectorize(bars):
    arr = np.zeros(len(bars))
    for i, bar in enumerate(bars):
        arr[i] = bar._raw['o'];
    return arr

# gets the different tech companies from the tech.csv
def getSymbols():

    #the source directory used is the one used from the program that runs it
    fileName = 'data/tech.csv'
    raw_data = open(fileName, 'rt')
    data = np.loadtxt(raw_data, usecols = (0), skiprows = 1, delimiter = ',', dtype = np.str)
    
    return data

# gets past data from the alpaca trade api and saves it to a file
def PastData(allSymbols=getSymbols()):
    dataLimit = 1000  # amount of samples

    data = None

    count = len(allSymbols) // 100 + 1
    for i in range(count):

        # ASKTOBY
        e = (i + 1) * 100
        symbols = allSymbols[i * 100: e].tolist()

        barset = api.get_barset(symbols, "1Min", limit=dataLimit)

        vectors = []

        for symbol in symbols:
            vector = vectorize(barset[symbol])
            if (len(vector) == dataLimit):
                vectors.append(vector)

        arr = np.row_stack(tuple(vectors))

        data = arr if (i == 0) else np.row_stack((data, arr))

        #print('progress:', str(i) + "/" + str(count - 1))

    np.savetxt("data/techData.csv", data, delimiter=',', fmt='%f')

#gets past data from alpaca trade api but returns the data instead of saving to a file
def PastData2(allSymbols=getSymbols()):
    dataLimit = 1000  # amount of samples

    data = None

    count = len(allSymbols) // 100 + 1
    for i in range(count):

        # ASKTOBY
        e = (i + 1) * 100
        symbols = allSymbols[i * 100: e].tolist()

        barset = api.get_barset(symbols, "1Min", limit=dataLimit)

        vectors = []

        for symbol in symbols:
            vector = vectorize(barset[symbol])
            if (len(vector) == dataLimit):
                vectors.append(vector)

        arr = np.row_stack(tuple(vectors))

        data = arr if (i == 0) else np.row_stack((data, arr))

        #print('progress:', str(i) + "/" + str(count - 1))

    dataDictionary = {}
    for x in range(0, len(data)):
        dataDictionary["AM." + allSymbols[x]] = data[0]

    return dataDictionary;


