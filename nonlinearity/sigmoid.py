import numpy as np
import csv
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


with open("linear_cw2_training.csv", "r") as i :
    rawdata = list(csv.reader(i, delimiter = ","))

sigmoid_data = np.array(rawdata[1:],dtype=np.float)

xdata = sigmoid_data[:, 6]
ydata = sigmoid_data[:, 3]

def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0)))+b
    return (y)

p0 = [max(ydata), np.median(xdata),1, min(ydata)]

popt, pcov = curve_fit(sigmoid, xdata, ydata, p0, method='dogbox')

plt.scatter(xdata, ydata, color='red')
plt.scatter(xdata, sigmoid(xdata, *popt))
plt.legend(['actual data', 'curve fit'])
plt.show()