import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

iris = sns.load_dataset("iris")
print(iris.head())

data = iris[["sepal_length", "petal_length", "sepal_width", "species"]]
data_df = data[(data["species"] == "setosa") | (data["species"] == "versicolor")]
print(data.head())

X = data_df[["sepal_length", "petal_length", "sepal_width"]]
y = data_df["species"]

data_df_setosa = data_df[data_df["species"] == "setosa"]
data_df_versicolor = data_df[data_df["species"] == "versicolor"]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

model = SVC(kernel="linear", C=100000)
model.fit(X, y)


w = model.coef_[0]
b = model.intercept_[0]


x1 = np.linspace(min(data_df["sepal_length"]), max(data_df["sepal_length"]), 11)
x2 = np.linspace(min(data_df["petal_length"]), max(data_df["petal_length"]), 11)
X1, X2 = np.meshgrid(x1, x2)


X3 = -(w[0] * X1 + w[1] * X2 + b) / w[2]

# граница разделения классов
ax.plot_surface(X1, X2, X3, alpha=0.7, color="gray")

# опорные вектора
ax.scatter(
    model.support_vectors_[:, 0],
    model.support_vectors_[:, 1],
    model.support_vectors_[:, 2],
    s=400,
    facecolors="none",
    edgecolors="black",
)


x1_p = np.linspace(min(data_df["sepal_length"]), max(data_df["sepal_length"]), 10)
x2_p = np.linspace(min(data_df["petal_length"]), max(data_df["petal_length"]), 10)
x3_p = np.linspace(min(data_df["sepal_width"]), max(data_df["sepal_width"]), 10)
X1_p, X2_p, X3_p = np.meshgrid(x1_p, x2_p, x3_p)

X_p = pd.DataFrame(
    np.vstack([X1_p.ravel(), X2_p.ravel(), X3_p.ravel()]).T,
    columns=["sepal_length", "petal_length", "sepal_width"],
)
y_p = model.predict(X_p)

X_p["species"] = y_p

X_p_setosa = X_p[X_p["species"] == "setosa"]
X_p_versicolor = X_p[X_p["species"] == "versicolor"]


ax.scatter(
    X_p_setosa["sepal_length"],
    X_p_setosa["petal_length"],
    X_p_setosa["sepal_width"],
    alpha=0.1,
    label="Predicted Setosa",
)
ax.scatter(
    X_p_versicolor["sepal_length"],
    X_p_versicolor["petal_length"],
    X_p_versicolor["sepal_width"],
    alpha=0.1,
    label="Predicted Versicolor",
)


ax.scatter(
    data_df_setosa["sepal_length"],
    data_df_setosa["petal_length"],
    data_df_setosa["sepal_width"],
    label="Actual Setosa",
)
ax.scatter(
    data_df_versicolor["sepal_length"],
    data_df_versicolor["petal_length"],
    data_df_versicolor["sepal_width"],
    label="Actual Versicolor",
)

ax.set_xlabel("Sepal Length")
ax.set_ylabel("Petal Length")
ax.set_zlabel("Sepal Width")
plt.legend()
plt.show()
