import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

# Define the dataframe reading it from a file as CSV
df = pd.read_csv('breast-cancer-wisconsin.data.txt')
# Since we know we have some missing data, we want to replace it with a really large number
# Which will be treated as an outlier
df.replace('?', -99999, inplace=True)
# Drop the ID column since it's irrelevant and would mess with the algorithm
df.drop(['id'], 1, inplace=True)

# Define features as X, label as y
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

# Shuffle the data and separate it into training and testing data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# Define the classifier to use
clf = neighbors.KNeighborsClassifier()
# Train the classifier
clf.fit(X_train, y_train)

# Test the data for accuracy
accuracy = clf.score(X_test, y_test)
print(accuracy)

# Define an example to predict
example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 4, 1]])
example_measures = example_measures.reshape(len(example_measures), -1)

# Predict the class
prediction = clf.predict(example_measures)
print(prediction)