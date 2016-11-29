########################################
#
# Random walk will show the affects of randomly 
# buying selling or holding at each time point.
# This is to be usedd as a baseline against 
# which our agent can be compared
#
########################################

import numpy as np
import matplotlib.pyplot as plt
import random as rand


# initilize the starting number of stocks and the starting bank ballance
START_BANK = 100
START_STOCK = 100


# testing paramaters to be tuned by user
MAX_SELL = 50
MIN_SELL = 1
TRIALS = 50


# load the training file
def loadData(file):
	return np.genfromtxt(file, delimiter=',', skip_header=1,
            skip_footer=1, names=['date', 'open', 'high', 'low', 'close', 'adj'])

# buys num_to_trade number of stocks and updatets bank ballance accordingly, debt is allowed
def buy(time_index, num_to_trade = 1):
	global stocks_held, bank_ballance
	stocks_held += num_to_trade
	bank_ballance -= num_to_trade*stock['open'][time_index]
	portfolio.append(stocks_held*stock['open'][time_index] + bank_ballance)


# sells num_to_trade number of stocks and updatets bank ballance accordingly, 
# selling more than you own is not permitted
def sell(time_index, num_to_trade = 1):
	global stocks_held, bank_ballance

	if stocks_held <= 0:
		hold(time_index)
	else:
		if ((stocks_held - num_to_trade) <= 0): 
			bank_ballance += stocks_held*stock['open'][time_index]
			stocks_held = 0
		else:
			stocks_held -= num_to_trade
			bank_ballance += num_to_trade*stock['open'][time_index]
		portfolio.append(stocks_held*stock['open'][time_index] + bank_ballance)


# appends the same value to ballance
def hold(time_index):
	portfolio.append(stocks_held*stock['open'][time_index] + bank_ballance)


def randomWalk(stock):
	global stocks_held, bank_ballance, portfolio
	stocks_held = START_STOCK
	bank_ballance = START_BANK
	portfolio = []

	for i in range(len(stock['open'])):
		rand_num = rand.randint(0,2)
		# rand_to_trade = 1
		rand_to_trade = rand.randint(MIN_SELL, MAX_SELL)
		#wtf? why does python not have switch statements?
		if rand_num == 0:
			print 'sell'
			sell(i, rand_to_trade)
		elif rand_num == 1:
			print 'hold'
			print stocks_held
			hold(i)
		elif rand_num == 2:
			print 'buy'
			buy(i, rand_to_trade)

	return portfolio

###########
# Main code
###########


# pick the data file to read from
stock = loadData('BA_2Y_14_15.csv')



# print portfolio
# plotData(stock['open'], portfolio)

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
plt.show()

