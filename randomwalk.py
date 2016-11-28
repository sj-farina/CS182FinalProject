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
stocks_held = 100
current_ballance = [0]



# plot the stock prices to compare to your bank balance (random walk)
def plotData(stock, ballance):
	fig = plt.figure()
	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)
	ax1.set_title("Title")    
	ax1.set_xlabel('time')
	ax1.set_ylabel('price')
	ax1.plot(range(len(stock)), stock, color='r', label='stock price')
	ax2.plot(range(len(ballance)), ballance, color='b', label='ballance')
	leg = ax1.legend()
	plt.show()

# load the training file
def loadData(file):
	return np.genfromtxt(file, delimiter=',', skip_header=1,
            skip_footer=1, names=['date', 'open', 'high', 'low', 'close', 'adj'])

# buys num_to_buy number of stocks and updatets bank ballance accordingly, debt is allowed
def buy(time_index, num_to_buy = 1):
	global stocks_held
	current_ballance.append(current_ballance[time_index] - stock['open'][time_index] * num_to_buy)
	stocks_held += num_to_buy

# sells num_to_sell number of stocks and updatets bank ballance accordingly, 
# selling more than you own is not permitted
def sell(time_index, num_to_sell = 1):
	global stocks_held
	# you can't sell more stocks than you have, hold instead
	if stocks_held <= 0:
		hold(time_index)
	else:
		# if selling ammount would make us negative, just sell everything
		if ((stocks_held - num_to_sell) <= 0): 
			current_ballance.append(current_ballance[time_index] + stock['open'][time_index] * stocks_held)
			stocks_held = 0
		else:
			current_ballance.append(current_ballance[time_index] + stock['open'][time_index] * num_to_sell)
			stocks_held -= num_to_sell
# appends the same value to ballance
def hold(time_index):
	current_ballance.append(current_ballance[time_index])



###########
# Main code
###########


# pick the data file to read from
stock = loadData('BA_2Y_14_15.csv')

for i in range(len(stock['open'])):
	rand_num = rand.randint(0,2)
	#wtf? why does python not have switch statements?
	if rand_num == 0:
		print 'sell'
		sell(i)
	elif rand_num == 1:
		print 'hold'
		hold(i)
	elif rand_num == 2:
		print 'buy'
		buy(i)

print current_ballance
plotData(stock['open'], current_ballance)



