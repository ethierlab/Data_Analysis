import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

x = [146, 160, 164, 170, 180, 200, 240, 260, 280, 300]
y = 0.075375, 0.10949999999999999 , 0.11775 , 0.20175 , 0.22820000000000001 , 0.32687499999999997 , 0.34299999999999997 , 0.34862499999999996 , 0.35424999999999995 , 0.35700000000000004


def sigmoid(x, x0, k):
    y = 1 / (1 + np.exp(-k * (x - x0)))
    return y

initial_guess = [0.01, 0]


# Perform the curve fitting
popt_sigmoid, pcov_sigmoid = curve_fit(sigmoid, x, y, p0=initial_guess)

# Generating y values based on the sigmoid fit
y_fit_sigmoid = sigmoid(x, *popt_sigmoid)

# Plot the original data and the sigmoid fit
plt.scatter(x, y, label='Original Data')
plt.plot(x, y_fit_sigmoid, 'r-', label='Fitted Sigmoid')

# Adding the legend to the plot
plt.legend()

# Display the plot
plt.show()
