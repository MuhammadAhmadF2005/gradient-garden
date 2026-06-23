import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# 1. Generate and scale synthetic data
X, y = make_blobs(n_samples=500, n_features=2, centers=3, random_state=42)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# 2. Initialize clusters
k = 3
clusters = {}
np.random.seed(23)

for idx in range(k):
    # Randomly initialize centers between -2 and 2
    center = 2 * (2 * np.random.random((X.shape[1],)) - 1)
    cluster = {"center": center, "points": []}
    clusters[idx] = cluster


# 3. Helper Functions
def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


# FIX 1: Defined the missing assign_clusters function
def assign_clusters(X, clusters):
    # CRITICAL: Clear previous points before re-assigning
    for i in range(k):
        clusters[i]["points"] = []

    for i in range(X.shape[0]):
        dist = []
        for j in range(k):
            dist.append(distance(X[i], clusters[j]["center"]))

        # Find the index of the closest center
        cur_cluster = np.argmin(dist)
        clusters[cur_cluster]["points"].append(X[i])

    return clusters


# FIX 2: Defined the missing update_clusters function
def update_clusters(X, clusters):
    for idx in range(k):
        points = np.array(clusters[idx]["points"])

        # Only update center if the cluster actually has points assigned to it
        if points.shape[0] > 0:
            new_center = points.mean(axis=0)
            clusters[idx]["center"] = new_center

    return clusters


def pred_cluster(X, clusters):
    pred = []
    for i in range(X.shape[0]):
        dist = []
        for j in range(k):
            dist.append(distance(X[i], clusters[j]["center"]))
        pred.append(np.argmin(dist))
    return pred


# 4. Execute one full iteration step of K-Means
clusters = assign_clusters(X, clusters)
clusters = update_clusters(X, clusters)
pred = pred_cluster(X, clusters)

# 5. Plot the result
plt.figure(figsize=(8, 6))
# Plot data points colored by their assigned cluster prediction
plt.scatter(X[:, 0], X[:, 1], c=pred, cmap="viridis", alpha=0.7, edgecolor="k")

# Plot final cluster centers as prominent red triangles
for i in clusters:
    center = clusters[i]["center"]
    plt.scatter(
        center[0],
        center[1],
        marker="^",
        c="red",
        s=150,
        edgecolor="black",
        label="Centroid" if i == 0 else "",
    )

plt.title("K-Means: After 1 Iteration Step")
plt.xlabel("Scaled Feature 1")
plt.ylabel("Scaled Feature 2")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.show()