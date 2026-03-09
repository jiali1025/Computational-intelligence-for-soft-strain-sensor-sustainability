import csv, numpy as np
from scipy.optimize import curve_fit


with open("linear_cw2_training.csv", "r") as f:
    raw = list(csv.reader(f))
data = np.array(raw[1:], dtype=float)
x = data[:, 6]
y = data[:, 3]

def sigmoid(x, L, x0, k, b):
    return L / (1 + np.exp(-k * (x - x0))) + b

p0 = np.array([y.max(), np.median(x), 1.0, y.min()], dtype=float)

seeds = [42, 99, 3407, 2023, 7]
results = []

for seed in seeds:
    rng = np.random.default_rng(seed)
    p0_perturb = p0 * (1.0 + rng.normal(0.0, 0.10, size=p0.shape))

    bounds = ([-np.inf, x.min()-10*np.std(x), -np.inf, -np.inf],
              [ np.inf, x.max()+10*np.std(x),  np.inf,  np.inf])

    popt, pcov = curve_fit(sigmoid, x, y, p0=p0_perturb, method='dogbox', bounds=bounds, maxfev=20000)

    y_hat = sigmoid(x, *popt)
    rss = float(np.sum((y - y_hat)**2))
    results.append((seed, popt, rss))

print(results)

