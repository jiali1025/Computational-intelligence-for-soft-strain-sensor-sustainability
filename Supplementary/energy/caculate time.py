import time
import numpy as np
import csv
from scipy.optimize import curve_fit

with open("linear_cw2_training.csv", "r") as i:
    rawdata = list(csv.reader(i, delimiter=","))

sigmoid_data = np.array(rawdata[1:], dtype=float)
xdata = sigmoid_data[:, 6]
ydata = sigmoid_data[:, 3]

def sigmoid(x, L ,x0, k, b):
    return L / (1 + np.exp(-k*(x-x0))) + b

p0 = [max(ydata), np.median(xdata), 1, min(ydata)]

N = 300
t_start = time.time()
for _ in range(N):
    popt, pcov = curve_fit(sigmoid, xdata, ydata, p0, method='dogbox', maxfev=20000)
t_end = time.time()

with open("run_window.txt", "w") as f:
    f.write(f"{t_start}\n{t_end}\n{N}\n")

print("start:", t_start, "end:", t_end, "N:", N)
