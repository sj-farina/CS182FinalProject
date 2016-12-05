########################################
# CS182 Final Project, Fall 2016
# Anita Xu & Janey Farina
#
# Random walk will show the affects of randomly 
# buying selling or holding at each time point.
# This is to be used as a baseline against 
# which our agent can be compared
#
########################################

import numpy as np
import matplotlib.pyplot as plt
import random as rd


# Initialize the starting number of stocks and the starting bank balance
START_BANK = 10000
START_STOCK = 0
# INFILE = 'BA_6M_15.csv'
# INFILE = 'BA_1Y_15.csv'
INFILE = 'BA_2Y_14_15.csv'
# INFILE = 'BA_5Y_11_15.csv'
# INFILE = 'BA_15Y_01_15.csv'


# testing parameters to be tuned by user
MAX_SELL = 50
MIN_SELL = 1
TRIALS = 10


# load the training file
def loadData(file):
	return np.genfromtxt(file, delimiter=',', skip_header=1,
            skip_footer=1, names=['date', 'open', 'high', 'low', 'close', 'adj'])

# buys num_to_trade number of stocks and updates bank balance accordingly, debt is allowed
def buy(time_index, num_to_trade = 1):
	global stocks_held, bank_balance
	if (num_to_trade*stock['open'][time_index] > bank_balance):
		num_to_trade = int(bank_balance)/int(stock['open'][time_index])
	stocks_held += num_to_trade
	bank_balance -= num_to_trade*stock['open'][time_index]
	portfolio.append(stocks_held*stock['open'][time_index] + bank_balance)


# sells num_to_trade number of stocks and updates bank balance accordingly, 
# selling more than you own is not permitted
def sell(time_index, num_to_trade = 1):
	global stocks_held, bank_balance

	if stocks_held <= 0:
		hold(time_index)
	else:
		if ((stocks_held - num_to_trade) <= 0): 
			bank_balance += stocks_held*stock['open'][time_index]
			stocks_held = 0
		else:
			stocks_held -= num_to_trade
			bank_balance += num_to_trade*stock['open'][time_index]
		portfolio.append(stocks_held*stock['open'][time_index] + bank_balance)


# appends the same value to balance
def hold(time_index):
	portfolio.append(stocks_held*stock['open'][time_index] + bank_balance)


def randomWalk(stock):
	global stocks_held, bank_balance, portfolio
	stocks_held = START_STOCK
	bank_balance = START_BANK
	portfolio = []

	for i in range(len(stock['open'])):
		rand_num = rd.randint(0,2)
		# rand_to_trade = 1
		rand_to_trade = rd.randint(MIN_SELL, MAX_SELL)
		#wtf? why does python not have switch statements?
		if rand_num == 0:
			sell(i, rand_to_trade)
		elif rand_num == 1:
			hold(i)
		elif rand_num == 2:
			buy(i, rand_to_trade)
		print(rand_num,stocks_held,bank_balance,portfolio[-1])
	return portfolio

###########
# Main code
###########


# Pick the data file to read from
stock = loadData(INFILE)

# Setting up the plot
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.set_title("Stock Value")    
ax1.set_xlabel('time')
ax1.set_ylabel('price')
ax2.set_title("Portfolio Value")    
ax2.set_xlabel('time')
ax2.set_ylabel('value')

ax1.plot(range(len(stock['open'])), stock['open'], color='r', label='Stock Price')

for each in range(TRIALS):
	rand_port = randomWalk(stock)
	ax2.plot(range(len(rand_port)), rand_port, label='Portfolio Value')

leg = ax1.legend()
# leg = ax2.legend()
plt.show()

