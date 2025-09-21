# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Step 2: Load and Prepare the Data
# Load the dataset from the CSV file
try:
    df = pd.read_csv('Mall_Customers.csv')
except FileNotFoundError:
    print("Error: 'Mall_Customers.csv' not found. Please make sure the dataset is in the same folder as your script.")
    exit()

# We only need 'Annual Income (k$)' and 'Spending Score (1-100)' for this analysis.
# Let's select these two columns for our clustering algorithm.
X = df.loc[:, ['Annual Income (k$)', 'Spending Score (1-100)']].values

# Step 3: Find the Optimal Number of Clusters (K) using the Elbow Method
# We'll calculate the Within-Cluster-Sum-of-Squares (WCSS) for different values of K.
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X)
    # The 'inertia_' attribute gives the WCSS value
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.grid(True)
plt.show()

# From the plot, the "elbow" is at K=5. This is our optimal number of clusters.

# Step 4: Build and Train the Final K-means Model
# We will use K=5 based on the Elbow Method.
optimal_k = 5
kmeans_model = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)

# Fit the model and predict the cluster for each data point
y_kmeans = kmeans_model.fit_predict(X)

# Step 5: Visualize the Customer Segments
plt.figure(figsize=(12, 8))

# Plot the data points for each of the 5 clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='violet', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='cyan', label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='orange', label='Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='red', label='Cluster 5')

# Plot the centroids of each cluster
plt.scatter(kmeans_model.cluster_centers_[:, 0], kmeans_model.cluster_centers_[:, 1], s=300, c='black', label='Centroids', marker='*')

plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()