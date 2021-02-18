import alpaca_trade_api as tradeapi
import config
import datetime

def connectToAccount():
    try:

        api = tradeapi.REST(config.API_KEY, config.SECRET_KEY, base_url='https://paper-api.alpaca.markets')
        api.list_positions()

        print("Account Accessed")

        return api, api.get_account()

    except tradeapi.rest.APIError:
        print("Invalid Keys")
        exit(1)

api, account = connectToAccount()


#can get past data

whichData = ["GME"]

barset = api.get_barset(whichData[0], "1Min", limit=450)
aapl_bars = barset[whichData[0]]
print(aapl_bars)