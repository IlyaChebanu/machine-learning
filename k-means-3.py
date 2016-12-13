import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')

# Some arbitrary points that can be easily formed into groups
X = np.array([[1, 2],
			  [1.5, 1.8],
			  [5, 8],
			  [8, 8],
			  [1, 0.6],
			  [9, 11]])

# List of colours
colors = ['g', 'r', 'c', 'b', 'k']

# Defining K_Means classifier class
class K_Means:
    # k -> how many clusters
    # tol -> how much the centroid is allowed to move (pct. change)
    # max_iter -> max amount of iterations for optimizing centroid position
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        self.centroids = {}
        # Iterate k times
        for i in range(self.k):
            # First 2 centroids will be the first 2 points in data (features)
            self.centroids[i] = data[i]

        # Begin optimization process
        for i in range(self.max_iter):
            # Keys: centroids
            # Values: feature sets
            self.classifications = {}
            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                # Creating a list being populated with k number of values
                # of distances from datapoint to the centroid
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                # Finding the mean of all the features
                # Re-define the new centroids
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)
            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                # If any of the centroids move more than the tollerance:
                # optimized => false
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tol:
                    optimized = False

            # If passed the optimization test, break out of the loop
            # Before max_iter
            if optimized:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification



clf = K_Means()
clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker="o", color="k", s=150, linewidths=5)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker='x', color=color, s=150, linewidths=5)

unknowns = np.array([[1, 3], [8, 9], [0, 3], [5, 4], [6, 4],])

for unknown in unknowns:
    classification = clf.predict(unknown)
    plt.scatter(unknown[0], unknown[1], marker="*", color=colors[classification], s=150, linewidths=5)

plt.show()
