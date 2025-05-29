import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans

iris = sns.load_dataset("iris")
data = iris[["sepal_length", "petal_length", "sepal_width", "species"]]
data_df = data[(data["species"] == "setosa") | (data["species"] == "versicolor")]


X = data_df[["sepal_length", "petal_length", "sepal_width"]]


kmeans = KMeans(n_clusters=2, random_state=42)
cluster_labels = kmeans.fit_predict(X)


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")


scatter1 = ax.scatter(
    X[cluster_labels == 0]["sepal_length"],
    X[cluster_labels == 0]["petal_length"],
    X[cluster_labels == 0]["sepal_width"],
    label="Cluster 0",
    alpha=0.8,
)
scatter2 = ax.scatter(
    X[cluster_labels == 1]["sepal_length"],
    X[cluster_labels == 1]["petal_length"],
    X[cluster_labels == 1]["sepal_width"],
    label="Cluster 1",
    alpha=0.8,
)

# центры кластеров
centers = kmeans.cluster_centers_
ax.scatter(
    centers[:, 0],
    centers[:, 1],
    centers[:, 2],
    s=200,
    marker="*",
    c="red",
    label="Cluster Centers",
)


ax.set_xlabel("Sepal Length")
ax.set_ylabel("Petal Length")
ax.set_zlabel("Sepal Width")
plt.title("К-средних кластеризация")


plt.legend()


print("\nЦентры кластеров:")
for i, center in enumerate(centers):
    print(f"\nCluster {i}:")
    print(f"Sepal Length: {center[0]:.2f}")
    print(f"Petal Length: {center[1]:.2f}")
    print(f"Sepal Width: {center[2]:.2f}")

# точность кластеризации по сравнению с фактическими данными
accuracy = np.mean(cluster_labels == (data_df["species"] == "versicolor").astype(int))
print(f"\nТочность кластеризации: {accuracy:.2%}")

plt.show()
