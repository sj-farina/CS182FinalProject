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


# Initialize the starting number of stocks and the starting bank balance
START_BANK = 100
START_STOCK = 100

# Training variables
EPSILON = .3
ALPHA = .5
DISCOUNT = .7
ITERATIONS = 10
LOOKAHEAD = 50

# Helpers and things
stocks_held = START_STOCK
bank_ballance = START_BANK
portfolio = []
qvalues = [] 


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
    # if bank_ballance <= 0:
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
    for i in range(span)

    pass

# Returns difference between current value and mean of last "span" points 
def meanDiff(cur_time, span):
    avg = 0.0
    for i in range(span):
        if (cur_time - i) > 0:
            avg += data_set[cur_time - i]
    avg = avg / span
    return data_set[cur_time] - avg




# Extract the features at the current time point
def getFeatures(cur_time):
    pass
    features{price} = data_set[cur_time]
    features{pointSlopeLocal} = pointSlope(cur_time, )
    features{avgSlopeLocal} = 
    features{meanDiff} = 





# Determine the q value from (weights (dot) features)
def getQValue():
    pass

    # qvalues = 0.0
    # features = getFeatures(cur_time, )
    # weights = 
    # for feature in features:
    #     qvalues += features[feature] * weights[feature]

    # return qvalues


# Update our the weights given a transition
def update(cur_time, action, reward):
    pass





########################################
# STOCK MANIPULATION CODE
########################################

# buys num_to_trade number of stocks and updates bank balance accordingly, debt is allowed
def buy(cur_time, num_to_trade = 10):
    global stocks_held, bank_ballance
    stocks_held += num_to_trade
    bank_ballance -= num_to_trade*data_set[cur_time]
    portfolio.append(stocks_held*data_set[cur_time] + bank_ballance)


# sells num_to_trade number of stocks and updates bank balance accordingly, 
def sell(cur_time, num_to_trade = 10):
    global stocks_held, bank_ballance
    if ((stocks_held - num_to_trade) <= 0): 
        bank_ballance += stocks_held*data_set[cur_time]
        stocks_held = 0
    else:
        stocks_held -= num_to_trade
        bank_ballance += num_to_trade*data_set[cur_time]
    portfolio.append(stocks_held*data_set[cur_time] + bank_ballance)

# appends the current value to balance
def hold(cur_time):
    portfolio.append(stocks_held*data_set[cur_time] + bank_ballance)


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
values = collections.Counter()
if (TRAINING):
    print 'Im training'
    # How many times should we run this?
    for i in range(ITERATIONS):
        # Iterates over array, time (cur_time) is arbitrary, two points per day
        for cur_time in range(LOOKAHEAD, len(data_set) - 2*LOOKAHEAD):
            # state = getShortTermTrend(cur_time)
            # nextState = getShortTermTrend(cur_time+1)
            action = pickAction(state, cur_time)
            reward = getReward(cur_time, action)
            update(cur_time, action, nextState, reward)

# TODO: Make this selectable from cmdline and save/load trained dataset elsewhere
    TRAINING = 0
    TESTING = 1
    print values


data_set = loadData(TESTFILE)


if (TESTING):
    print 'im testing'
    stocks_held = START_STOCK
    bank_ballance = START_BANK
    portfolio = []
    for cur_time in range(len(data_set)):
        state = getShortTermTrend(cur_time)
        action = getBestAction(state, cur_time)
        tradeStocks(cur_time, action)

else:
    print "Please Specify Training -train or Testing -test"


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





