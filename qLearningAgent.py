########################################
# CS182 Final Project, Fall 2016
# Anita Xu & Janey Farina
#
# Q-Learning Agent for 182 final project
# Version 1
# Code modeled after CS182 Pset 3
# 
########################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections  as mc
import random as rd
import math
import collections 

# Pick the file to read from
INFILE = 'BA_6M_15.csv'
# INFILE = 'BA_1Y_15.csv'
# INFILE = 'BA_2Y_14_15.csv'
# INFILE = 'BA_5Y_11_15.csv'
# INFILE = 'BA_15Y_01_15.csv'


# Initialize the starting number of stocks and the starting bank balance
START_BANK = 100
START_STOCK = 100

# Training variables
EPSILON = 1
ALPHA = 1
DISCOUNT = 1
ITERATIONS = 10

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


# Find the slope between this point and the last
def getShortTermTrend(cur_time):
    # if this is the first data point, assume a slope of zero
    if cur_time == 0:
        return 0
    # Multiply by 100, set to int, equiv of truncating at 1 decimals
    slope = int((data_set[cur_time] - data_set[cur_time-1]) * 10)
    # Cap -10 to 10 to limit state space
    if slope > 10:
        return 10
    if slope < -10:
        return -10
    return rlope

# Determine which actions are available given stocks held and bank balance
def getLegalActions(cur_time):
    # For the simplest case, lets just say all actions are always valid
    return [buy, sell, hold]

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
        return data_set[cur_time + 5] - data_set[cur_time] 
    elif action == 'sell':
        return -(data_set[cur_time + 5] - data_set[cur_time])
    elif action == 'hold':
        return 0
    
# Pick the action to take based on epsilon and best action
def pickAction(state):
    legalActions = getLegalActions()  
    if (rd.random() < EPSILON):
        return rd.choice(legalActions)
    return getBestAction(state)

# Determine the best possible action based on the stored values information 
def getBestAction(cur_time):
    pass
# Determine the best possible "score" from a given state
def getMaxStateValue():
    pass


# Update our qvalue array
def update(state, action, nextState, reward):
    values[state,action] = values[state,action] + ALPHA * (reward +
        DISCOUNT * computeValueFromQValues(nextState) - values[state,action])





########################################
# MAIN CODE 
########################################

data_set = loadData(INFILE)
values = collections.Counter()
# How many times should we run this?
for i in range(ITERATIONS):
    # Iterates over array, time (cur_time) is arbitrary, two points per day
    for cur_time in range(len(data_set)):
        state = getShortTermTrend(cur_time)
        nextState = getShortTermTrend(cur_time+1)
        action = getBestAction(cur_time)
        reward = getReward(cur_time, action)
        update(state, action, nextState, reward)




# # Optional plot for reference
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(range(len(data_set)), data_set, color='r', label='Stock Price')
# plt.show()





