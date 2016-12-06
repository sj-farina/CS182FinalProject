########################################
# CS182 Final Project, Fall 2016
# Anita Xu & Janey Farina
#
# Q-Learning Agent for 182 final project
# Code modeled after CS182 Pset 3
#
########################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections  as mc
import random as rd
import math, collections, sys

# Pick the file to read from
# INFILE = 'BA_6M_15.csv'
# INFILE = 'BA_1Y_15.csv'
# INFILE = 'BA_2Y_14_15.csv'
# INFILE = 'BA_5Y_11_15.csv'
INFILE = 'BA_15Y_01_13.csv'


# Initialize the starting number of stocks and the starting bank balance
START_BANK = 10000
START_STOCK = 0

# Helpers and things
stocks_held = START_STOCK
bank_balance = START_BANK
portfolio = []
qvalues = [] 


# Load the training file

def loadData(file):
    file = np.genfromtxt(file, delimiter=',', skip_header=1,
            skip_footer=1, names=['date', 'open', 'high', 'low', 'close', 'adj'])
    open_value = file['open']
    close_value = file['close']
    # Zips the opening and closing values into one array
    return close_value


########################################
# Q LEARNING CODE 
########################################

# Find the slope between this point and the last
def getShortTermTrend(cur_time):
    # # if this is the first data point, assume a slope of zero
    # if cur_time == 0:
    #     return 0
    # Multiply by 100, set to int, equiv of truncating at 1 decimals
    slope = int((data_set[cur_time] - data_set[cur_time - LOOKAHEAD])*5)
    # Cap -10 to 10 to limit state space
    if slope > 50:
        return 50
    if slope < -50:
        return -50
    return slope

# Determine which actions are available given stocks held and bank balance
def getLegalActions(cur_time):
    # For the simplest case, lets just say all actions are always valid
    return ['buy', 'sell', 'hold']

    # If you have no $$ and no stocks, you can't do anything
    if bank_balance <= 0:
        if stocks_held <= 0:
            return [hold]
        return [sell, hold]
    elif stocks_held <= 0:
        return [buy, hold]
    else:
        return [buy, sell, hold]

# Determine the reward we get in a given state given the action
# Reward is the difference between current portfolio and next portfolio
def getReward(cur_time, action):
    if action == 'buy':
        return data_set[cur_time + LOOKAHEAD] - data_set[cur_time] 
    elif action == 'sell':
        return -(data_set[cur_time + LOOKAHEAD] - data_set[cur_time])
    elif action == 'hold':
        return 0
    
# Pick the action to take based on epsilon and best action
def pickAction(state, cur_time):
    legalActions = getLegalActions(cur_time)  
    if (rd.random() < EPSILON):
        return rd.choice(legalActions)
    return getBestAction(state, cur_time)

# Determine the best possible action based on the stored values information 
def getBestAction(state, cur_time):
    bestScore = -sys.maxint - 1
    # Default to hold as the most neutral choice
    bestAction = 'hold'

    for action in getLegalActions(cur_time):
        score = values[state,action]
        if score > bestScore:
            bestScore = score
            bestAction = action

    return bestAction


# Determine the best possible "score" from a given state
def getMaxStateValue(state, cur_time):
    bestScore = -sys.maxint - 1
    for action in getLegalActions(cur_time):
        score = values[state,action]
        if score > bestScore:
            bestScore = score
    return bestScore


# Update our qvalue array
def update(cur_time, state, action, nextState, reward):
    values[state,action] = values[state,action] + ALPHA * (reward +
        DISCOUNT * getMaxStateValue(nextState, cur_time +1) - values[state,action])


########################################
# STOCK MANIPULATION CODE
########################################

# buys num_to_trade number of stocks and updates bank balance accordingly, debt is allowed
def buy(cur_time, num_to_trade = 10):
    global stocks_held, bank_balance
    if num_to_trade*data_set[cur_time] > bank_balance:
        num_to_trade = int(bank_balance)/int(data_set[cur_time])
    stocks_held += num_to_trade
    bank_balance -= num_to_trade*data_set[cur_time]
    portfolio.append(stocks_held*data_set[cur_time] + bank_balance)


# sells num_to_trade number of stocks and updates bank balance accordingly, 
def sell(cur_time, num_to_trade = 10):
    global stocks_held, bank_balance
    if ((stocks_held - num_to_trade) <= 0): 
        bank_balance += stocks_held*data_set[cur_time]
        stocks_held = 0
    else:
        stocks_held -= num_to_trade
        bank_balance += num_to_trade*data_set[cur_time]
    portfolio.append(stocks_held*data_set[cur_time] + bank_balance)

# appends the current value to balance
def hold(cur_time):
    portfolio.append(stocks_held*data_set[cur_time] + bank_balance)


def tradeStocks(cur_time, action):
    if action == 'sell':
        sell(cur_time)
    elif action == 'hold':
        hold(cur_time)
    elif action == 'buy':
        buy(cur_time)




########################################
# MAIN CODE 
########################################
# Training variables

EPSILON = 0.05
alpha = [0.05,0.2,0.5,1]
DISCOUNT = 0.8
ITERATIONS = 100
LOOKAHEAD = 20

# Optional plot for reference
fig = plt.figure()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)

LIMIT_training = 3269 # train on data 2001-2013, test on 2014-2015
INFILE = 'BA_15Y_01_15.csv'
data_set = loadData(INFILE)

for i in alpha:
    ALPHA = i
    
    values = collections.Counter()
    
    print 'Im training'
    # How many times should we run this?
    for i in range(ITERATIONS):
        # Iterates over array, time (cur_time) is arbitrary, two points per day
        for cur_time in range(LOOKAHEAD, LIMIT_training - LOOKAHEAD):
            state = getShortTermTrend(cur_time)
            nextState = getShortTermTrend(cur_time+1)
            action = pickAction(state, cur_time)
            reward = getReward(cur_time, action)
            update(cur_time, state, action, nextState, reward)
    # print values


        # print values
    # INFILE = 'BA_6M_15.csv'
    # INFILE = 'BA_1Y_15.csv'
    # INFILE = 'BA_2Y_14_15.csv'
    # data_set = loadData(INFILE)


# # # TODO: Make this selectable from cmdline and save/load trained dataset elsewhere
# #     print 'in', values
# #     f = open('valueFile', 'w')
# #     f.write(str(values))

# # # INFILE = 'BA_6M_15.csv'
# # # INFILE = 'BA_1Y_15.csv'
# # # INFILE = 'BA_2Y_14_15.csv'


# # elif (TESTING):
# #     s = open('valueFile', 'r').read()
# #     values = eval(s)
# #     print 'out', values
#     data_set = loadData(INFILE)

    print 'im testing'
    stocks_held = START_STOCK
    bank_balance = START_BANK
    portfolio = []
    for cur_time in range(LIMIT_training,len(data_set)):
        state = getShortTermTrend(cur_time)
        action = getBestAction(state, cur_time)
        tradeStocks(cur_time, action)
        print (state,action,stocks_held,bank_balance,portfolio[-1])
    ax2.plot(range(len(portfolio)), portfolio, label='Portfolio Value')

stock = 'BA_2Y_14_15.csv'
stockdata = loadData(stock)
ax1.plot(range(len(stockdata)), stockdata, color='r', label='Stock Price')
ax2.legend(alpha, loc='best')
# plt.show()

########################################
# DISPLAY CODE 
########################################



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
        # print(rand_num,stocks_held,bank_balance,portfolio[-1])
    return portfolio

###########
# Main code
###########


# Pick the data file to read from
stock = loadData(INFILE)

# Setting up the plot
# fig = plt.figure()
# ax1 = fig.add_subplot(211)
ax3 = fig.add_subplot(313)
ax3.set_title("Stock Value")    
ax3.set_xlabel('time')
ax3.set_ylabel('price')
ax3.set_title("Portfolio Value")    
ax3.set_xlabel('time')
ax3.set_ylabel('value')

for each in range(TRIALS):
    rand_port = randomWalk(stock)
    ax3.plot(range(len(rand_port)), rand_port, label='Portfolio Value')

# leg = ax3.legend()
# leg = ax2.legend()
plt.show()





