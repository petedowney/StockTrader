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
        vectors3 = []

        for symbol in symbols:
            vector = vectorize(barset[symbol], 'o')
            vector2 = vectorize(barset[symbol], 'v')
            vector3 = vectorize(barset[symbol], 'h') - vectorize(barset[symbol], 'l')

            if len(vector) == dataLimit:
                vectors.append(vector)
                vectors2.append(vector2)
                vectors3.append(vector3)

        arr1 = np.row_stack(tuple(vectors))
        arr2 = np.row_stack(tuple(vectors2))
        arr3 = np.row_stack(tuple(vectors3))

        data_o = arr1 if (i == 0) else np.row_stack((data_o, arr1))
        data_v = arr2 if (i == 0) else np.row_stack((data_v, arr2))
        data_r = arr2 if (i == 0) else np.row_stack((data_r, arr3))

        #finalData = np.stack((dataO, dataV))
        #print('progress:', str(i) + "/" + str(count - 1))

    np.savetxt("data/techDataO.csv", data_o, delimiter=',', fmt='%f')
    np.savetxt("data/techDataV.csv", data_v, delimiter=',', fmt='%f')
    np.savetxt("data/techDataR.csv", data_r, delimiter=',', fmt='%f')

#gets past data from alpaca trade api but returns the data instead of saving to a file
def PastData2(api, allSymbols=getSymbols(), dataLimit=1000):

    data = None

    count = len(allSymbols) // 100 + 1

    for i in range(count):

        e = (i + 1) * 100
        symbols = allSymbols[i * 100: e].tolist()

        barset = api.get_barset(symbols, "1Min", limit=dataLimit)

        vectors = []
        vectors2 = []
        vectors3 = []

        for symbol in symbols:
            vector = vectorize(barset[symbol], 'o')
            vector2 = vectorize(barset[symbol], 'v')
            vector3 = vectorize(barset[symbol], 'h') - vectorize(barset[symbol], 'l')

            if len(vector) == dataLimit:
                vectors.append(vector)
                vectors2.append(vector2)
                vectors3.append(vector3)

        arr1 = np.row_stack(tuple(vectors))
        arr2 = np.row_stack(tuple(vectors2))
        arr3 = np.row_stack(tuple(vectors3))

        data_o = arr1 if (i == 0) else np.row_stack((data_o, arr1))
        data_v = arr2 if (i == 0) else np.row_stack((data_v, arr2))
        data_r = arr2 if (i == 0) else np.row_stack((data_r, arr3))

        data = np.stack((data_o, data_v, data_r))

        #print('progress:', str(i) + "/" + str(count - 1))

    dataDictionary = {}

    for x in range(0, data.shape[1]):
        dataDictionary["AM." + allSymbols[x]] = data[:, x, :]

    return dataDictionary;


