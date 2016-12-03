from math import sqrt
import numpy as np
from collections import Counter
import warnings
import pandas as pd
import random


def k_nearest_neighbors(data, predict, k=3):
	if len(data) >= k:
		warnings.warn('K is set to a value less than total voting groups')

	distances = []
	# Each group is a key, eg. 'k', 'r'
	for group in data:
		# Each feature is the point, eg. data[k][0] = 1, 2
		for features in data[group]:
			# Get the euclidean distance between each point and the point we want to classify (predict)
			euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
			# append the distance, group pair to the distances array
			distances.append([euclidean_distance, group])
	 
	# Create a list of k groups based on the distance
	# The closest k amount of elements will be in the list
	votes = [i[1] for i in sorted(distances)[:k]]
	# Count the votes and find the 1 most common vote
	# the [0][0] is necessary since the most_common function returns a list of a tuple with the group name and count
	vote_result = Counter(votes).most_common(1)[0][0]
	confidence = Counter(votes).most_common(1)[0][1] / k

	return vote_result, confidence

# Import the csv data
df = pd.read_csv('breast-cancer-wisconsin.data.txt')
# Replace question marks with outlier data
df.replace('?', -99999, inplace=True)
# Drop the IDs to not mess with the predictions
df.drop(['id'], 1, inplace=True)
# Convert the data to floats and a list of lists
# Must convert to floats first as some of the data is treated as a string
full_data = df.astype(float).values.tolist()
# Shuffle the data. random.shuffle shuffles the data in place
random.shuffle(full_data)

# Define how many % of the data to be the testing data
test_size = 0.2
train_set = {2: [], 4: []}
test_set = {2: [], 4: []}
# Training data is the rest of the % (80% in our case)
train_data = full_data[:-int(test_size*len(full_data))]
# Testing data is the last 20% of the data
test_data = full_data[-int(test_size*len(full_data)):]

# Populate the dictionaries
for i in train_data:
	# append the features to the train_set keys, which are the label
	train_set[i[-1]].append(i[:-1])
for i in test_data:
	# do the same for the testing data
	test_set[i[-1]].append(i[:-1])

# Creating a counter to count the amount of correct predictions out of the total set
correct = 0
total = 0
# Here group is the label (2 or 4)
for group in test_set:
	# data is all of the features
	for data in test_set[group]:
		vote, confidence = k_nearest_neighbors(train_set, data, k=5)
		# if the algorithm voted for the correct group, increment correct counter
		if group == vote:
			correct += 1
		total += 1

print('Accuracy: {}'.format(correct / total))