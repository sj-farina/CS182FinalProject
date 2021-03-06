########################################
# CS182 Final Project, Fall 2016
# Anita Xu & Janey Farina
#
# Approximate Q-Learning Agent for 182 final project
# Code modeled after CS182 Pset 3 (from UC Berkeley, http://ai.berkeley.edu)
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


# Tune these values for different k, n, and statespace
# Local span = k, lookahead = n
LOCAL_SPAN = 5
LOOKAHEAD = 1
NUM_FEATS = 2 # 2 features
# The +/- cutoff for state space
CUTOFF = 20


# Initialize the starting number of stocks and the starting bank balance
START_BANK = 10000
START_STOCK = 0
default = 10 # default buy or sell 10 stocks

# Training variables
EPSILON = .2
ALPHA = .8
DISCOUNT = 0
ITERATIONS = 10


########################################
# MISC
########################################

# Helpers and things
stocks_held = START_STOCK
bank_balance = START_BANK
portfolio = []
features = [0]*NUM_FEATS


# Load the training file, return closing data
def loadData(file):
    file = np.genfromtxt(file, delimiter=',', skip_header=1,
            skip_footer=1, names=['date', 'open', 'high', 'low', 'close', 'adj'])
    close_value = file['close']
    return close_value


########################################
# Q LEARNING CODE 
########################################

# Determine which actions are available given stocks held and bank balance
def getLegalActions(cur_time):
    # For the simplest case, we just say all actions are always valid and reassess in buy() and sell()
    return ['buy', 'sell', 'hold']

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
def yesterdaySlope(cur_time, action):
    slope = float((data_set[cur_time] - data_set[cur_time - 1])*1.0/data_set[cur_time - 1])
    feat1= 0
    if action == 'buy':
        feat1 = slope * (stocks_held + default)/1000.0
    elif action == 'hold':
        feat1 = slope * (stocks_held)/1000.0
    else:
        feat1 = slope * (stocks_held - default)/1000.0
    return feat1

# Returns average slope in past span days
def avgSlope(cur_time, span, action):
    avg = 0.0
    for i in range(span):
        if (cur_time - i) > 0:
            avg += (data_set[cur_time - i] - data_set[cur_time - i - 1])*1.0/data_set[cur_time - i - 1]
    slope = (avg*1.0 / span)
    feat2 = 0
    if action == 'buy':
        feat2 = slope * (stocks_held + default)/1000.0
    elif action == 'hold':
        feat2 = slope * (stocks_held)/1000.0
    else:
        feat2 = slope * (stocks_held - default)/1000.0
    return feat2

# Returns difference between current value and mean of last "span" points 
def meanDiff(cur_time, span,action):
    avg = 0.0
    for i in range(span):
        if (cur_time - i) > 0:
            avg += data_set[cur_time - i]
    avg = avg*1.0 / span
    slope= data_set[cur_time] - avg
    feat3 = 0
    if action == 'buy':
        feat3 = slope * (stocks_held + default)/1000.0
    elif action == 'hold':
        feat3 = slope * (stocks_held)/1000.0
    else:
        feat3 = slope * (stocks_held - default)/1000.0
    return feat3

# Returns an array of all implemented features
def getFeatures(cur_time,action):
    global features
    features[0] = yesterdaySlope(cur_time, action)
    features[1] = avgSlope(cur_time, LOCAL_SPAN,action)
    # features[2] = meanDiff (cur_time, RUNNING_SPAN,action)
    return features

# Determine the q value from (weights (dot) features)
def getQValue(cur_time, action):
    features = getFeatures(cur_time,action)
    qval = 0.0
    for i in range(len(features)):
        qval += weights[i] * features[i]
    return qval

# Return the max possible Q value given a state
def maxQValue(cur_time):
    bestScore = -sys.maxint -1
    for action in ['buy', 'hold', 'sell']:
        score = getQValue(cur_time, action)
        if score > bestScore:
            bestScore = score
    return bestScore
# Returns the best possible action given a state
def getBestAction(cur_time):
    bestScore = -sys.maxint -1
    bestAction = 'hold'
    for action in ['buy', 'hold', 'sell']:
        score = getQValue(cur_time, action)
        if score > bestScore:
            bestScore = score
            bestAction = action
    return bestAction

# Update our the weights given a transition
def update(cur_time, action, reward):
    features = getFeatures(cur_time,action)
    # update using Approximate Q-learning equations in Lecture 11
    difference = reward + DISCOUNT * maxQValue(cur_time +1) - getQValue(cur_time, action)
    for i in range(len(weights)):
        weights[i] += ALPHA * difference * features[i]
        if weights[i]>CUTOFF:
            weights[i] = CUTOFF
        elif weights[i] < -CUTOFF:
            weights[i]= -CUTOFF



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

total = 0 
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

# # To test multiple runs on the same files, uncomment next line and indent up to line 286
# for times in range(50):

# print 'Im training'
weights = [0]*NUM_FEATS
for i in range(ITERATIONS):
    # Iterates over array, time (cur_time) is arbitrary, two points per day
    for cur_time in range(LOOKAHEAD, LIMIT_training-LOOKAHEAD):
        state= [data_set[cur_time],stocks_held]
        action = pickAction(cur_time)
        reward = getReward(cur_time, action)
        update(cur_time, action, reward)

# TESTFILE = 'BA_2Y_14_15.csv'
# data_set = loadData(TESTFILE)


# print 'im testing'
# print weights
stocks_held = START_STOCK
bank_balance = START_BANK
portfolio = []
store_actions =[]
for cur_time in range(LIMIT_training,len(data_set)):
    action = getBestAction(cur_time)
    tradeStocks(cur_time, action)
    # print(action, stocks_held, bank_balance, portfolio[-1])
    temp= 0
    if action == 'buy':
        temp = 1
    elif action == 'sell':
        temp = -1
    store_actions.append(temp)
# total += portfolio[-1] - portfolio[0]
print portfolio[-1] - portfolio[0]
# print weights
ax2.plot(range(len(portfolio)), portfolio, label='Portfolio Value')

np.savetxt("BA_approx_actions.csv",store_actions, delimiter=",")
np.savetxt("BA_approx_port.csv",portfolio, delimiter=",")
# print "average =", total*1.0/1.0
print INFILE, "Approx- Q"



########################################
# DISPLAY CODE 
########################################

# Optional plot for reference
# change the file as needed
stock = 'BA_2Y_14_16.csv'
stockdata = loadData(stock)
ax1.plot(range(len(stockdata)), stockdata, color='r', label='Stock Price')


# ax1.axis([0,600,0,160])
# ax2.axis([0,600,0,np.max(portfolio)])
plt.show()





