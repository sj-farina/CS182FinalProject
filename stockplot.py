import numpy as np
import matplotlib.pyplot as plt

def plotData(data):
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.set_title("Title")    
	ax1.set_xlabel('time')
	ax1.set_ylabel('price')
	ax1.plot(range(len(data)), data, color='r', label='stock price')
	leg = ax1.legend()
	plt.show()



stock = np.genfromtxt('BA_6M_15.csv', delimiter=',', skip_header=10,
                     skip_footer=10, names=['date', 'open', 'high', 'low', 'close', 'adj'])

plotData(stock['high'])

