########################################
# CS182 Final Project, Fall 2016
# Anita Xu & Janey Farina
#
# Q-Learning Agent for 182 final project
# Code adapted from CS182 Pset 3
#
########################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections  as mc
import random as rd
import math, collections, sys

########################################
# EDIT THIS CODE TO TUNE PARAMETERS
########################################

# Pick the file to read from
# INFILE = 'KSS_16Y_00_16.csv'
INFILE = 'BA_16Y_00_16.csv'

# Our datafile is one large file, this picks the cutoff point between testing and training
LIMIT_training = 3522 # train on data 2001-2013, test on 2014-2015

# Training Variables
EPSILON = 0.2
ALPHA = 0.8
DISCOUNT = 0
ITERATIONS = 100

# lookahead gives the reward
LOOKAHEAD = 5
# lookback gives the slope
LOOKBACK = 2

# Initialize the starting number of stocks and the starting bank balance
START_BANK = 10000
START_STOCK = 0

# testing parameters for random agent
MAX_SELL = 50
MIN_SELL = 1
TRIALS = 10


########################################
# MISC
########################################

# Helpers and counters
stocks_held = START_STOCK
bank_balance = START_BANK
portfolio = []
qvalues = [] 

# Load the training file
def loadData(file):
    file = np.genfromtxt(file, delimiter=',', skip_header=1,
            skip_footer=1, names=['date', 'open', 'high', 'low', 'close', 'adj'])
    close_value = file['close']
    return close_value


########################################
# Q LEARNING CODE 
########################################

# Find the slope between this point and the last
def getShortTermTrend(cur_time):
    # # if this is the first data point, assume a slope of zero
    # Multiply by 100, set to int, equiv of truncating at 1 decimals
    slope = int((data_set[cur_time] - data_set[cur_time - LOOKBACK])*5)
    # Cap -10 to 10 to limit state space
    if slope > 10:
        return 10
    if slope < -10:
        return -10
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

# Randomly buys sells or holds at each time point, this is our baseline agent to beat
def randomWalk(stock):
    global stocks_held, bank_balance, portfolio
    stocks_held = START_STOCK
    bank_balance = START_BANK
    portfolio = []
    store_array=[]
    for i in range(len(stock)):
        rand_num = rd.randint(-1,1)
        # rand_to_trade = 1
        rand_to_trade = rd.randint(MIN_SELL, MAX_SELL)
        if rand_num == -1:
            sell(i, rand_to_trade)
        elif rand_num == 0:
            hold(i)
        elif rand_num == 1:
            buy(i, rand_to_trade)
        # print(rand_num,stocks_held,bank_balance,portfolio[-1])
        store_array.append(rand_num)
    # np.savetxt("random_actions.csv",store_array, delimiter=",")
    # np.savetxt("random_port.csv",portfolio, delimiter=",")
    return portfolio


########################################
# MAIN CODE 
########################################

# Optional plot for reference
fig = plt.figure()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)


total = 0
data_set = loadData(INFILE) 
values = collections.Counter()

# Train for ITERATTIONS numer of times
# print 'Im training'
for i in range(ITERATIONS):
    # Iterates over array, time (cur_time) is arbitrary, one point per day
    for cur_time in range(LOOKBACK, LIMIT_training - LOOKAHEAD):
        state = getShortTermTrend(cur_time)
        nextState = getShortTermTrend(cur_time+1)
        action = pickAction(state, cur_time)
        reward = getReward(cur_time, action)
        update(cur_time, state, action, nextState, reward)
# print values

# Test on a new set of data
# print 'im testing'
stocks_held = START_STOCK
bank_balance = START_BANK
portfolio = []
store_actions =[]
for cur_time in range(LIMIT_training,len(data_set)):
    state = getShortTermTrend(cur_time)
    action = getBestAction(state, cur_time)
    tradeStocks(cur_time, action)
    # print (state,action,stocks_held,bank_balance,portfolio[-1])
    temp= 0
    if action == 'buy':
        temp = 1
    elif action == 'sell':
        temp = -1
    store_actions.append(temp)
ax2.plot(range(len(portfolio)), portfolio, label='Portfolio Value')
print portfolio[-1] - portfolio[0]
total += portfolio[-1] - portfolio[0]

########################################
# DISPLAY CODE 
########################################

stock = 'BA_2Y_14_16.csv'
stockdata = loadData(stock)
ax1.plot(range(len(stockdata)), stockdata, color='r', label='Stock Price')
print INFILE, "Q-learning"
print "average of 50 trials =", total/50.0
# ax2.legend(alpha, loc='best')
# plt.show()

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

rand_total = 0
for each in range(50):
    rand_port = randomWalk(data_set)
    ax3.plot(range(len(rand_port)), rand_port, label='Portfolio Value')
#     print rand_port[-1] - rand_port[0]
    rand_total += rand_port[-1]-10000
    print rand_port[-1]-10000
# print stock[-1][1] -  stock[0][1]
# leg = ax3.legend()
# leg = ax2.legend()
plt.show()

print "average_random=", rand_total/50.0



