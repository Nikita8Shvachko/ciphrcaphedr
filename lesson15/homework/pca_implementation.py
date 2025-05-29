import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

iris = sns.load_dataset("iris")
data = iris[["sepal_length", "petal_length", "sepal_width", "species"]]
data_df = data[(data["species"] == "setosa") | (data["species"] == "versicolor")]


X = data_df[["sepal_length", "petal_length", "sepal_width"]]
y = data_df["species"]


data_df_setosa = data_df[data_df["species"] == "setosa"]
data_df_versicolor = data_df[data_df["species"] == "versicolor"]


pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)


print("\nКомпоненты PCA (как исходные признаки влияют на каждый PC):")
feature_names = ["sepal_length", "petal_length", "sepal_width"]
for i, component in enumerate(pca.components_):
    print(f"\nPC{i+1}:")
    for feature, weight in zip(feature_names, component):
        print(f"{feature}: {weight:.3f}")


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")


ax.scatter(
    data_df_setosa["sepal_length"],
    data_df_setosa["petal_length"],
    data_df_setosa["sepal_width"],
    label="Setosa",
    alpha=0.8,
)
ax.scatter(
    data_df_versicolor["sepal_length"],
    data_df_versicolor["petal_length"],
    data_df_versicolor["sepal_width"],
    label="Versicolor",
    alpha=0.8,
)


mean_point = X.mean(axis=0)

# Построение векторов PCA
for i, component in enumerate(pca.components_):
    scaled_component = component * np.std(X, axis=0)
    ax.quiver(
        mean_point[0],
        mean_point[1],
        mean_point[2],
        scaled_component[0],
        scaled_component[1],
        scaled_component[2],
        color=["r", "g", "b"][i],
        label=f"PC{i+1}",
        arrow_length_ratio=0.1,
    )


ax.set_xlabel("Sepal Length")
ax.set_ylabel("Petal Length")
ax.set_zlabel("Sepal Width")

plt.legend()

# explained variance
print("\nExplained variance:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {ratio:.3f}")

plt.show()
