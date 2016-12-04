import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans

style.use('ggplot')

# Some arbitrary points that can be easily formed into groups
X = np.array([[1, 2],
			  [1.5, 1.8],
			  [5, 8],
			  [8, 8],
			  [1, 0.6],
			  [9, 11]])

# Define the classifier to use
clf = KMeans(n_clusters=2)
# Train the classifier
clf.fit(X)

# Get the coordinates for the centroids
centroids = clf.cluster_centers_
# Get the label IDs for colouring
labels = clf.labels_

# List of colours
colors = ['g.', 'r.', 'c.', 'b.', 'k.']

# For each feature, plot the point and colour it with the label colour
for i in range(len(X)):
	plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=25)
# Plot the centroids with an x
plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=150, linewidths=5)
# Draw the plot
plt.show()