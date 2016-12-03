from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

# Arbitrary X and Y data for testing
# xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
# ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

# A function to generate a dataset with hm being how many data points
# variance being the variance on the y axis, lower variance should mean higher r^2
def create_dataset(hm, variance, step=2, correlation=False):
	val = 1
	ys = []
	for i in range(hm):
		y = val + random.randrange(-variance, variance)
		ys.append(y)
		if correlation and correlation == 'pos':
			val += step
		elif correlation and correlation == 'neg':
			val -= step
	xs = [i for i in range(hm)]

	return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

# Defining a function that calculates the best fit slope
# Using this equation:
#
#      mean(x) * mean(y) - mean(x * y)
# m = ---------------------------------
#         mean(x) ** 2 - mean(x ** 2)
#
#
# Equation for the y intercept:
#
# b = mean(y) - m * mean(x)
#
def best_fit_slope_and_intercept(xs, ys):
	m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
			((mean(xs) ** 2) - mean(xs ** 2)))
	b = mean(ys) - m * mean(xs)
	return m, b

# To calculate the squared error we need the actual points, and the points on the line
# Squared error is the difference between points on the line and the original points, squared
def squared_error(ys_orig, ys_line):
	return sum((ys_line - ys_orig) ** 2)

# Formula for calculating R squared (coefficient of determination)
# se stands for squared error
# y hat stands for regression line
#              ^
#           se(y)
# r^2 = 1 - _____
#              _
#           se(y)
def coefficient_of_determination(ys_orig, ys_line):
	# Create an array of points for the mean line using the original y points
	y_mean_line = [mean(ys_orig) for y in ys_orig]
	squared_error_regr = squared_error(ys_orig, ys_line)
	squared_error_y_mean = squared_error(ys_orig, y_mean_line)
	return 1 - (squared_error_regr / squared_error_y_mean)
	
# Generate the dataset for testing
# With variance 80, r^2 around 0.2
# With variance 40, r^2 averaged about 0.5
# With variance 10, r^2 around 0.93
xs, ys = create_dataset(40, 80, 2, correlation='pos')

# Create an array using the formula y = mx + b
# containing the y values for each x
m, b = best_fit_slope_and_intercept(xs, ys)
regression_line = [(m * x) + b for x in xs]


r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

plt.scatter(xs, ys)
plt.plot(xs, regression_line)
plt.show()