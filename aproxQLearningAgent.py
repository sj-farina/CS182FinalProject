########################################
# CS182 Final Project, Fall 2016
# Anita Xu & Janey Farina
#
# Approximate Q-Learning Agent for 182 final project
# Code modeled after CS182 Pset 3
#
########################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections  as mc
import random as rd
import math, collections, sys

# Are you training or testing???
TRAINING = 1
TESTING = 0

# Pick the file to read from
# INFILE = 'BA_6M_15.csv'
# INFILE = 'BA_1Y_15.csv'
# INFILE = 'BA_2Y_14_15.csv'
# INFILE = 'BA_5Y_11_15.csv'
INFILE = 'BA_15Y_01_15.csv'

TESTFILE = 'BA_1Y_15.csv'

# Feature 
RUNNING_SPAN = 200
LOCAL_SPAN = 10
NUM_FEATS = 4


# Initialize the starting number of stocks and the starting bank balance
START_BANK = 10000
START_STOCK = 0

# Training variables
EPSILON = .3
ALPHA = .5
DISCOUNT = .7
ITERATIONS = 10
LOOKAHEAD = 50

# Helpers and things
stocks_held = START_STOCK
bank_balance = START_BANK
portfolio = []
features = [0]*NUM_FEATS


# Load the training file
def loadData(file):
    file = np.genfromtxt(file, delimiter=',', skip_header=1,
            skip_footer=1, names=['date', 'open', 'high', 'low', 'close', 'adj'])
    open_value = file['open']
    close_value = file['close']
    # Zips the opening and closing values into one array
    return np.insert(close_value, np.arange(len(open_value)), open_value)


########################################
# Q LEARNING CODE 
########################################

# Determine which actions are available given stocks held and bank balance
def getLegalActions(cur_time):
    # For the simplest case, lets just say all actions are always valid
    return ['buy', 'sell', 'hold']

    # # If you have no $$ and no stocks, you can't do anything
    # if bank_balance <= 0:
    #     if stocks_held <= 0:
    #         return [hold]
    #     return [sell, hold]
    # elif stocks_held <= 0:
    #     return [buy, hold]
    # else:
    #     return [buy, sell, hold]

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
def pickAction(cur_time):
    legalActions = getLegalActions(cur_time)  
    if (rd.random() < EPSILON):
        return rd.choice(legalActions)
    return getBestAction(cur_time)


# Find the slope between this point and the last
def pointSlope(cur_time, span):
    # # if this is the first data point, assume a slope of zero
    # if cur_time == 0:
    #     return 0

    # Multiply by 100, set to int, equiv of truncating at 2 decimals
    slope = int((data_set[cur_time] - data_set[cur_time - span]) * 100)
    # Cap -10 to 10 to limit state space
    if slope > 10:
        return 10
    if slope < -10:
        return -10
    return slope

# Returns average
def avgSlope(cur_time, span):
    avg = 0.0
    for i in range(span):
        if (cur_time - i) > 0:
            avg += data_set[cur_time - i] - data_set[cur_time - i - 1]
    return (avg / span)


# Returns difference between current value and mean of last "span" points 
def meanDiff(cur_time, span):
    avg = 0.0
    for i in range(span):
        if (cur_time - i) > 0:
            avg += data_set[cur_time - i]
    avg = avg / span
    return data_set[cur_time] - avg

def getFeatures(cur_time):
    global features
    features[0] = 0
    features[1] = pointSlope(cur_time, LOCAL_SPAN)
    features[2] = avgSlope(cur_time, LOCAL_SPAN)
    features[3] = meanDiff (cur_time, RUNNING_SPAN)

    return features

# Determine the q value from (weights (dot) features)
def getQValue(cur_time, action):
    features = getFeatures(cur_time)
    qval = 0.0
    for i in range(len(features)):
        qval += weights[i] * features[i]
    print 'weights', weights
    # print 'feat', features
    # print 'qval', qval
    return qval

def maxQValue(cur_time):
    bestScore = -sys.maxint -1
    bestAction = 'hold'
    for action in ['buy', 'hold', 'sell']:
        score = getQValue(cur_time, action)
        if score > bestScore:
            bestScore = score
            bestAction = action
    return bestScore

def getBestAction(cur_time):
    bestScore = -sys.maxint -1
    bestAction = 'hold'
    for action in ['buy', 'hold', 'sell']:
        score = getQValue(cur_time, action)
        # print score
        if score > bestScore:
            bestScore = score
            bestAction = action
    return bestAction

# def getBestAction(cur_time):
#     return maxQValue(cur_time)[1]

# Update our the weights given a transition
def update(cur_time, action, reward):
    features = getFeatures(cur_time)
    print 'reward', reward
    print 'discount', DISCOUNT
    print 'maxq', maxQValue(cur_time +1)
    print 'qval', getQValue(cur_time, action)
    difference = reward + DISCOUNT * maxQValue(cur_time +1) - getQValue(cur_time, action)
    for i in range(len(weights)):
        weights[i] += ALPHA * difference * features[i]
        if weights[i]>20:
            weights[i] = 20
        elif weights[i] < -20:
            weights[i]= -20




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

data_set = loadData(INFILE)
weights = [0]*NUM_FEATS
print weights 
if (TRAINING):
    print 'Im training'
    # How many times should we run this?
    for i in range(ITERATIONS):
        # Iterates over array, time (cur_time) is arbitrary, two points per day
        for cur_time in range(LOOKAHEAD, len(data_set) - 2*LOOKAHEAD):
            action = pickAction(cur_time)
            reward = getReward(cur_time, action)
            update(cur_time, action, reward)

# TODO: Make this selectable from cmdline and save/load trained dataset elsewhere
    TRAINING = 0
    TESTING = 1
    # print values

data_set = loadData(TESTFILE)


if (TESTING):
    print 'im testing'
    stocks_held = START_STOCK
    bank_balance = START_BANK
    portfolio = []
    for cur_time in range(len(data_set)):
        action = getBestAction(cur_time)
        tradeStocks(cur_time, action)
        print(action, bank_balance, stocks_held, portfolio[-1])


print weights



########################################
# DISPLAY CODE 
########################################

# Optional plot for reference
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.plot(range(len(data_set)), data_set, color='r', label='Stock Price')
ax2.plot(range(len(portfolio)), portfolio, label='Portfolio Value')

plt.show()





