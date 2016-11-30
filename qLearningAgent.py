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
import random,math

# Pick the file to read from
INFILE = 'BA_1Y_15.csv'

# Initialize the starting number of stocks and the starting bank balance
START_BANK = 100
START_STOCK = 100



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
    # multiply by 100, set to int, equiv of truncating at 2 decimals
    return int((data_set[cur_time] - data_set[cur_time-1]) * 100)

# Determine which actions are available given stocks held and bank balance
def getLegalActions(cur_time):
    # If you have no $$ and no stocks, you can't do anything
    if bank_ballance <= 0:
        if stocks_held <= 0:
            return [hold]
        return [sell, hold]
    elif stocks_held <= 0:
        return [buy, hold]
    else:
        return [buy, sell, hold]

# Determine the reward we get in a given state given the action
def getReward(cur_time, action):
    pass
# Pick the action to take based on epsilon and best action
def pickAction():
    pass
# Determine the best possible action based on the stored values information 
def getBestAction():
    pass
# Determine the best possible "score" from a given state
def getMaxStateValue():
    pass
# Update our qvalue array
def update():
    pass




########################################
# MAIN CODE
########################################

data_set = loadData(INFILE)


# Iterates over array, time (i) is arbitrary
for i in range(len(data_set)):
    print i, ' = ', getShortTermTrend(i)




# Optional plot for reference
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(range(len(data_set)), data_set, color='r', label='Stock Price')
plt.show()





