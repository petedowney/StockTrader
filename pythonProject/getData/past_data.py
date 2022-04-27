import math
import sys

import numpy as np
import pandas as pd
from datetime import date
from datetime import timedelta
from datetime import datetime


def vectorize(bars, dataType):
    arr = np.zeros(len(bars))
    for i, bar in enumerate(bars):
        arr[i] = bar.get(dataType);
    return arr

# gets the different tech companies from the tech.csv
def get_symbols():
    # the source directory used is the one used from the program that runs it
    file_name = 'data/tech.csv'
    raw_data = open(file_name, 'rt')
    data = np.loadtxt(raw_data, usecols=0, skiprows=1, delimiter=',', dtype=np.str)

    return data


# calculates the exponential moving average
def calculate_ema(data, smoothing):
    ema = np.zeros((data.shape))

    for company in range(0, data.shape[0]):
        ema[company, 0] = data[company, 0] * (smoothing / 1)
        for time_stamp in range(1, data.shape[1]):
            ema[company, time_stamp] = \
                data[company, time_stamp] * (smoothing / 1001) + \
                ema[company, time_stamp - 1] * (1 - (smoothing / 1001))

    return ema


# def get_bars(symbols)

# gets past data from the alpaca trade api and saves it to a file
def get_past_data_to_csv(api, allSymbols=get_symbols(), dataLimit=1000):
    print(api)
    data = None

    count = len(allSymbols) // 100 + 1
    for i in range(count):

        # ASKTOBY
        e = (i + 1) * 100
        symbols = allSymbols[i * 100: e].tolist()

        barset = api.get_bars(symbols, "1Min", limit=10000)

        # TODO generify this
        vectors = []
        vectors2 = []
        vectors3 = []

        for symbol in symbols:
            vector = vectorize(barset[symbol]._raw, 'o')
            vector2 = vectorize(barset[symbol]._raw, 'v')
            vector3 = vectorize(barset[symbol]._raw, 'h') - vectorize(barset[symbol]._raw, 'l')

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

        # finalData = np.stack((dataO, dataV))
        # print('progress:', str(i) + "/" + str(count - 1))

    np.savetxt("data/techDataO.csv", data_o, delimiter=',', fmt='%f')
    np.savetxt("data/techDataV.csv", data_v, delimiter=',', fmt='%f')
    np.savetxt("data/techDataR.csv", data_r, delimiter=',', fmt='%f')
    np.savetxt("data/techDataEMA.csv", data_ema, delimiter=',', fmt='%f')

# interpolates the data by date for each minute
def interpolate(data):

    dates = pd.to_datetime(data[:, 0])
    timestamp = 0
    while timestamp < len(dates) - 1:

        # checks if there is a minute bar that is one minute ahead one timestamp ahead
        if not (dates[timestamp].minute + 1) + dates[timestamp].hour * 60 == \
               dates[timestamp + 1].minute + dates[timestamp + 1].hour * 60 and \
                (dates[timestamp].minute + 1) + dates[timestamp].hour * 60 < \
                dates[timestamp + 1].minute + dates[timestamp + 1].hour * 60:
            data = np.insert(data, timestamp + 1, data[timestamp], axis=0)
            dates = np.insert(dates, timestamp + 1, dates[timestamp] + timedelta(minutes=1))
        timestamp += 1

    return data


# gets past data from alpaca trade api but returns the data instead of saving to a file
def get_past_data(api, all_symbols=get_symbols()):

    print("?")
    # I hate all of this code with a burning passion
    # the way the old API worked was so much better

    # TODO clean up code
    # TODO should I make a class for barset

    count = math.ceil(len(all_symbols) / 25)

    barset_dictionary = dict.fromkeys(all_symbols, [[]])

    for i in range(count):

        # will break up the companies into lists of whatever the multiplier is
        e = (i + 1) * 25
        symbols = all_symbols[i * 25: e].tolist()

        curr_date = datetime.utcnow()
        # if the market has not opened for the day sets the day back one
        if curr_date.timestamp() < curr_date.replace(hour=13, minute=30).timestamp():
            curr_date = curr_date - timedelta(days=1)

        # TODO this should really be the for loop on the outside
        for day in range(7):

            # finds the next day that the market is open
            # starts from today (or yesterday) and goes backwords from their
            new_date = str(curr_date.date()) == \
                           api.get_calendar((curr_date).isoformat("T") + "Z",
                                            (curr_date).isoformat("T") + "Z")[0]._raw["date"]

            while not new_date:
                curr_date = curr_date - timedelta(days=1)
                new_date = str(curr_date.date()) == \
                           api.get_calendar((curr_date).isoformat("T") + "Z",
                                            (curr_date).isoformat("T") + "Z")[0]._raw["date"]

            start_date = curr_date.replace(hour=13, minute=30)
            end_date = curr_date.replace(hour=20, minute=00)

            # makes sure that the list of companies given by symbols is not empty
            if len(symbols) > 0:

                if datetime.utcnow().timestamp() < end_date.timestamp() and str(start_date.date()) == str(datetime.utcnow().date()):
                    # gets the bar of today
                    barset = api.get_bars(symbols, "1Min", start=start_date.isoformat("T") + "Z",
                                          end=(datetime.utcnow() - timedelta(minutes=15, seconds=1)).isoformat("T") + "Z",
                                          limit=10000)
                else:
                    # gets the barset if it is not today or the market has already closed for today
                    barset = api.get_bars(symbols, "1Min", start=start_date.isoformat("T") + "Z",
                                          end=end_date.isoformat("T") + "Z", limit=10000)

                # iterates through the barset and adds it to the barset_dictionary
                for n in range(0, len(barset)):
                    company_name = barset[n]._raw["S"]

                    if barset_dictionary[company_name][0] == []:
                        barset_dictionary[company_name] = \
                            np.array([[barset[n]._raw["t"],
                                       barset[n]._raw["o"],
                                       barset[n]._raw["v"],
                                       barset[n]._raw["h"] -
                                       barset[n]._raw["l"]]])
                        #barset_dictionary[company_name][0] = str(barset[n]._raw["t"])
                    else:
                        new_bar_array = [[barset[n]._raw["t"],
                                          barset[n]._raw["o"],
                                          barset[n]._raw["v"],
                                          barset[n]._raw["h"] -
                                          barset[n]._raw["l"]]]
                        barset_dictionary[company_name] = np.append(new_bar_array,
                                                            barset_dictionary[company_name], axis=0)
            curr_date = curr_date - timedelta(days=1)

    print("?2")
    for company in barset_dictionary.keys():
        barset_dictionary[company] = np.sort(barset_dictionary[company], axis=0)
        # TODO fix interpolation
        #barset_dictionary[company] = interpolate(barset_dictionary[company])

    print("converting")
    # convets dictionary to an array
    # TODO make sure that array does not contain timestamps
    data = [[[]]]
    non_1000 = []
    for company in all_symbols:

        if barset_dictionary[company] != [[]] and barset_dictionary[company].shape[0] > 1000:
            if data == [[[]]]:
                data = [barset_dictionary[company][-1000:, 1:]]
            else:
                data = np.append(data, [barset_dictionary[company][-1000:, 1:]], axis=0)
        else:
            non_1000 = np.append(non_1000, company)

    data = np.swapaxes(data, 1, 2).astype(float)

    # (company, channels, timestamps)

    data_ema = calculate_ema(np.squeeze(data[:, 0]), 2)

    data = np.append(data, np.expand_dims(data_ema, 1), axis=1)
    print("converting done")
    return data, non_1000
