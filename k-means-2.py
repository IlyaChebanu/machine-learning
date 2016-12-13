import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd

style.use('ggplot')

# Read excel spreadsheet with the data into a dataframe using pandas
df = pd.read_excel('titanic.xls')
# Drop the body number and name columns
# I found that leaving just class, sex, fare and boat gives best results
df.drop(['body', 'name', 'embarked', 'sibsp', 'parch', 'ticket', 'home.dest', 'cabin', 'age'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
print(df.head())

def handle_non_numberical_data(df):
    columns = df.columns.values

    for column in columns:
        # Create a dictionary that will store all the int values
        # For the non-numeric keys
        text_digit_vals = {}

        # Define a function that retrieves the int values of keys
        def convert_to_int(val):
            # Simply return the value of the dictionary key: eg. 'Female': 0
            return text_digit_vals[val]

        # As we're going through each column,
        # Check the datatype of the current column
        # If the column isn't of numeric type
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            # Convert all column values to a list
            column_contents = df[column].values.tolist()
            # Convert that list to a set, so that we have no repetitions
            unique_elements = set(column_contents)
            # Initialise an int value for the non-numeric key
            x = 0
            # Loop through each unique element
            for unique in unique_elements:
                # If the element is not in the dictionary
                if unique not in text_digit_vals:
                    # Add it to the dictionary and set value to the index int
                    text_digit_vals[unique] = x
                    # Increment for next unique element
                    x += 1

            # Reset the values of column by mapping the function to value in col
            df[column] = list(map(convert_to_int, df[column]))

    # At the end of the function simply return the new dataframe
    return df

df = handle_non_numberical_data(df)
# print(df.head())

# Assign the features and label
X = preprocessing.scale(X)
y = np.array(df['survived'])

# Create a KMeans classifier with 2 clusters (since only 2 possible outcomes)
clf = KMeans(n_clusters=2)
# Train the classifier
clf.fit(X)

# Test the classifier and calculate the accuracy
correct = 0
for i in range(len(X)):
    # Take a single row of features as float
    predict_me = np.array(X[i].astype(float))
    # Reshape the row
    predict_me = predict_me.reshape(-1, len(predict_me))
    # Try and predict the label
    prediction = clf.predict(predict_me)
    # Test if correct
    if prediction[0] == y[i]:
        # If correct, increment counter
        correct += 1

print(correct / len(X))
