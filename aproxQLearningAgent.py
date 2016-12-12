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

# Pick the file to read from
# INFILE = 'BA_6M_15.csv'
# INFILE = 'BA_1Y_15.csv'
# INFILE = 'BA_2Y_14_15.csv'
# INFILE = 'BA_5Y_11_15.csv'
# INFILE = 'KSS_16Y_00_16.csv'

# Feature 
# RUNNING_SPAN = 20
LOCAL_SPAN = 5
NUM_FEATS = 2


# Initialize the starting number of stocks and the starting bank balance
START_BANK = 10000
START_STOCK = 0
default = 10 # default buy or sell 10 stocks

# Training variables
EPSILON = .2
ALPHA = .8
DISCOUNT = 0
ITERATIONS = 10
LOOKAHEAD = 1

# Helpers and things
stocks_held = START_STOCK
bank_balance = START_BANK
portfolio = []
features = [0]*NUM_FEATS


# Load the training file
def loadData(file):
    file = np.genfromtxt(file, delimiter=',', skip_header=1,
            skip_footer=1, names=['date', 'open', 'high', 'low', 'close', 'adj'])
    close_value = file['close']
    # Zips the opening and closing values into one array
    return close_value


########################################
# Q LEARNING CODE 
########################################

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
def pickAction(cur_time):
    legalActions = getLegalActions(cur_time)  
    if (rd.random() < EPSILON):
        return rd.choice(legalActions)
    return getBestAction(cur_time)


# Find the slope between this point and the last
def yesterdaySlope(cur_time, action):
    # # if this is the first data point, assume a slope of zero
    # if cur_time == 0:
    #     return 0

    # Multiply by 100, set to int, equiv of truncating at 2 decimals
    slope = float((data_set[cur_time] - data_set[cur_time - 1])*1.0/data_set[cur_time - 1])
    # Cap -10 to 10 to limit state space
    feat1= 0
    if action == 'buy':
        feat1 = slope * (stocks_held + default)/1000.0
    elif action == 'hold':
        feat1 = slope * (stocks_held)/1000.0
    else:
        feat1 = slope * (stocks_held - default)/1000.0
    # if feat1 > 10:
    #     return 10
    # if feat1 < -10:
    #     return -10
    return feat1

# Returns average
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
        if score > bestScore:
            bestScore = score
            bestAction = action
    return bestAction

# def getBestAction(cur_time):
#     return maxQValue(cur_time)[1]

# Update our the weights given a transition
def update(cur_time, action, reward):
    features = getFeatures(cur_time,action)
    # print 'reward', reward
    # print 'qval', getQValue(cur_time, action)
    # print 'weights', weights
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
INFILE = 'BA_15Y_01_15.csv'
data_set = loadData(INFILE)

LIMIT_training = 3270 # train on data 2001-2013, test on 2014-2015
total = 0 
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
for times in range(50):
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
    total += portfolio[-1] - portfolio[0]
    print portfolio[-1] - portfolio[0]
    # print weights
    ax2.plot(range(len(portfolio)), portfolio, label='Portfolio Value')

    np.savetxt("BA_approx_actions.csv",store_actions, delimiter=",")
    np.savetxt("BA_approx_port.csv",portfolio, delimiter=",")
print "average =", total*1.0/50
print INFILE, "Approx- Q"
########################################
# DISPLAY CODE 
########################################

# Optional plot for reference


stock = 'BA_2Y_14_15.csv'
stockdata = loadData(stock)
ax1.plot(range(len(stockdata)), stockdata, color='r', label='Stock Price')
# ax1.axis([0,600,0,160])
# ax2.axis([0,600,0,np.max(portfolio)])
plt.show()





