import alpaca_trade_api as tradeapi
import numpy as np

# ASKTOBY
def vectorize(bars, dataType):
    arr = np.zeros(len(bars))
    for i, bar in enumerate(bars):
        arr[i] = bar._raw[dataType];
    return arr

# gets the different tech companies from the tech.csv
def getSymbols():

    #the source directory used is the one used from the program that runs it
    fileName = 'data/tech.csv'
    raw_data = open(fileName, 'rt')
    data = np.loadtxt(raw_data, usecols=0, skiprows=1, delimiter=',', dtype=np.str)
    
    return data

# gets past data from the alpaca trade api and saves it to a file
def PastData(api, allSymbols=getSymbols(), dataLimit=1000):

    data = None

    count = len(allSymbols) // 100 + 1
    for i in range(count):

        # ASKTOBY
        e = (i + 1) * 100
        symbols = allSymbols[i * 100: e].tolist()

        barset = api.get_barset(symbols, "1Min", limit=dataLimit)

        vectors = []
        vectors2 = []

        for symbol in symbols:
            vector = vectorize(barset[symbol], 'o')
            vector2 = vectorize(barset[symbol], 'v')
            if (len(vector) == dataLimit):
                vectors.append(vector)
                vectors2.append(vector2)

        arr1 = np.row_stack(tuple(vectors))
        arr2 = np.row_stack(tuple(vectors2))

        dataO = arr1 if (i == 0) else np.row_stack((dataO, arr1))
        dataV = arr2 if (i == 0) else np.row_stack((dataV, arr2))

        #finalData = np.stack((dataO, dataV))
        #print('progress:', str(i) + "/" + str(count - 1))

    np.savetxt("data/techDataO.csv", dataO, delimiter=',', fmt='%f')
    np.savetxt("data/techDataV.csv", dataV, delimiter=',', fmt='%f')

#gets past data from alpaca trade api but returns the data instead of saving to a file
def PastData2(api, allSymbols=getSymbols(), dataLimit=1000):

    data = None

    count = len(allSymbols) // 100 + 1

    for i in range(count):

        # ASKTOBY
        e = (i + 1) * 100
        symbols = allSymbols[i * 100: e].tolist()


        barset = api.get_barset(symbols, "1Min", limit=dataLimit)

        vectors = []
        vectors2 = []

        for symbol in symbols:
            vector = vectorize(barset[symbol], "o")
            vector2 = vectorize(barset[symbol], "v")
            if (len(vector) == dataLimit):
                vectors.append(vector)
                vectors2.append(vector2)

        arr = np.row_stack(tuple(vectors))
        arr2 = np.row_stack(tuple(vectors2))

        dataO = arr if (i == 0) else np.row_stack((dataO, arr))
        dataV = arr2 if (i == 0) else np.row_stack((dataV, arr2))

        data = np.stack((dataO, dataV))

        #print('progress:', str(i) + "/" + str(count - 1))

    dataDictionary = {}

    for x in range(0, data.shape[1]):
        dataDictionary["AM." + allSymbols[x]] = data[:,x,:]

    return dataDictionary;


