# Predicting the future Google stock prices using Linear Regression

import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

# Re-train the classifier?
train = False

# Get the dataframe from quandl
df = quandl.get("WIKI/GOOGL", authtoken="o_dsrgif3_EyBgzHK1sf")

# Re-create dataframe to just be these cols
# Open - Start of day price
# Close - End of day price
# Low - Lowest price of day
# High - Highest price of day
# Volume - Amount trades occured
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# Define a new col for High - low percent volatility
# (High - Low)
# ------------  * 100
#     Low
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0

# Define a new col for percent change from start of day to end of day
# (Close - Open)
# -------------- * 100
#      Open
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

# Define a new dataframe
# Only the cols we care about
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

# Defining the forecast column to be the Adj. Close
forecast_col = 'Adj. Close'

# Fill all NaN data with -99999
# -99999 will be treated as an outlier
# Allows us to not have to delete any data since NaN data can't be used
df.fillna(-99999, inplace=True)

# Number of days out
# Trying to predict out 1% of the dataframe
forecast_out = int(math.ceil(0.01 * len(df)))

# Defining the label column to be the forecast column
# Shifting the columns negatively (up)
# The label column for each row will now be the Adj. Close price 1% into the future
df['label'] = df[forecast_col].shift(-forecast_out)

# Features = capital X
# Labels = lower case y

# Features -> everything except label col
# df.drop() returns new dataframe, then converts to numpy array
# scale() scales values to be from -1 to 1
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])

# This is a rather strange one. If this is scaled, the accuracy for svn.SVR goes up
# But, the values are way off and makes it impossible to make a prediction
# y = preprocessing.scale(y)

# Prepare the training and testing data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

if train:
	# Define a classifier
	clf = LinearRegression(n_jobs=10)
	# Fit = train
	clf.fit(X_train, y_train)

	# Saving the classifier
	with open('linearregression.pickle', 'wb') as f:
		pickle.dump(clf, f)

# Load the saved classifier
pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

# Score = test
accuracy = clf.score(X_test, y_test)

# print("Predicting {} days into the future".format(forecast_out))
# print("Prediction accuracy: {}".format(accuracy))

# Predict the 31 future days
forecast_set = clf.predict(X_lately)

# print(forecast_set, accuracy, forecast_out)

# Fill the forecast column with numpy NaN values
df['Forecast'] = np.nan

# Get the new dates
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

# Populate dataframe with new dates and forecast values
for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	# df.loc references the index of the dataframe
	# list comp sets the Adj. Close, HL_PCT, PCT_change, Adj. Volume and label to NaN
	# Since we don't know what those values will be in the future
	# i is the forecast so + [i] adds the values to the forecast column
	df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]


# Plot the graph of the price with the future prediction
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# print(df['Forecast'])