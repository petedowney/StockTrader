import alpaca_trade_api as tradeapi
import numpy as np


# ASKTOBY
def vectorize(bars, dataType):
    arr = np.zeros(len(bars))
    for i, bar in enumerate(bars):
        arr[i] = bar._raw[dataType];
    return arr


# gets the different tech companies from the tech.csv
def get_symbols():

    #the source directory used is the one used from the program that runs it
    file_name = 'data/tech.csv'
    raw_data = open(file_name, 'rt')
    data = np.loadtxt(raw_data, usecols=0, skiprows=1, delimiter=',', dtype=np.str)
    
    return data

def calculate_ema(data, smoothing):

    ema = np.zeros((data.shape))

    for company in range(0, data.shape[0]):
        ema[company, 0] = data[company, 0] * (smoothing / 1)
        for time_stamp in range(1, data.shape[1]):
            ema[company, time_stamp] = \
                data[company, time_stamp] * (smoothing / (time_stamp + 1)) +\
                ema[company, time_stamp - 1] * (1 - (smoothing / (time_stamp + 1)))

    return ema



# gets past data from the alpaca trade api and saves it to a file
def get_past_data_to_csv(api, allSymbols=get_symbols(), dataLimit=1000):

    data = None

    count = len(allSymbols) // 100 + 1
    for i in range(count):

        # ASKTOBY
        e = (i + 1) * 100
        symbols = allSymbols[i * 100: e].tolist()

        barset = api.get_barset(symbols, "1Min", limit=dataLimit)

        # TODO generify this
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
        data_r = arr3 if (i == 0) else np.row_stack((data_r, arr3))
        data_ema = calculate_ema(data_o, 2)

        #finalData = np.stack((dataO, dataV))
        #print('progress:', str(i) + "/" + str(count - 1))

    np.savetxt("data/techDataO.csv", data_o, delimiter=',', fmt='%f')
    np.savetxt("data/techDataV.csv", data_v, delimiter=',', fmt='%f')
    np.savetxt("data/techDataR.csv", data_r, delimiter=',', fmt='%f')
    np.savetxt("data/techDataEMA.csv", data_ema, delimiter=',', fmt='%f')

# gets past data from alpaca trade api but returns the data instead of saving to a file
def get_past_data(api, all_symbols=get_symbols(), dataLimit=1000):

    data = None

    count = len(all_symbols) // 100 + 1

    for i in range(count):

        e = (i + 1) * 100
        symbols = all_symbols[i * 100: e].tolist()

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
        data_ema = calculate_ema(data_o, 2)

        data = np.stack((data_o, data_v, data_r, data_ema))

    return data


