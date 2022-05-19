# StockTrader

This program will automatically paper trade stocks in the tech sector. It uses Alpaca Markets to function

## Setup
1. Go to https://alpaca.markets and create a paper trading account
2. In the paper trading acccount generate a public and private key

> <img width="360" alt="Screen Shot 2022-05-18 at 8 14 13 PM" src="https://user-images.githubusercontent.com/55404565/169176071-a98ae87a-bd0f-4371-aac9-be1875f11d9a.png"> 

> <img width="359" alt="Screen Shot 2022-05-18 at 8 15 07 PM" src="https://user-images.githubusercontent.com/55404565/169176142-d315b5be-6f95-43e3-85a2-891ea76ea55a.png">


3. Download and open the project
4. Install the projects external libraries (keras, matplotlib, alpaca-trade-api, tensorflow, )
5. Create a new file in the getData folder called *config.py*

> <img width="211" alt="Screen Shot 2022-05-18 at 8 04 37 PM" src="https://user-images.githubusercontent.com/55404565/169174739-36cb5bca-9bac-4ad4-92ac-c101cb71caa3.png">

6. in the config.py file create two strings. One called **API_KEY** and the other called **SECRET_KEY**. Set these equal to the keys given by the Alpaca Acount

> <img width="461" alt="Screen Shot 2022-05-18 at 8 25 53 PM" src="https://user-images.githubusercontent.com/55404565/169177018-a9bc20ed-8e2c-4070-a328-f2e434d1c3ba.png">

7. Setup should be complete

## Use
In order for the best results the program should be launched at 9:16 on days that the market is open.
It will take around 15 minutes from run time for the program to be fully operational



