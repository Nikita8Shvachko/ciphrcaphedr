# Todo Убрать из данных ирис часть точек на которых мы обучаемся и убедиться что на предсказание влияет только опорные вектора


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC


iris = sns.load_dataset("iris")
data = iris[["sepal_length", "petal_length", "species"]]
data_df = data[(data["species"] == "setosa") | (data["species"] == "versicolor")]

X = data_df[["sepal_length", "petal_length"]]
y = data_df["species"]

fig, axes = plt.subplots(2, 2, figsize=(6, 6))
axes = axes.ravel()


def get_reduced_dataset(X, y, test_size, support_vectors, random_state=42):
    sv_df = pd.DataFrame(support_vectors, columns=["sepal_length", "petal_length"])

    # Выбираем опорные вектора
    is_sv = np.zeros(len(X), dtype=bool)
    for i, row in X.iterrows():
        is_sv[i] = any(
            (sv_df["sepal_length"] == row["sepal_length"])
            & (sv_df["petal_length"] == row["petal_length"])
        )

    non_sv_mask = ~is_sv
    X_non_sv = X[non_sv_mask]
    y_non_sv = y[non_sv_mask]

    n_to_keep = int(len(X_non_sv) * (1 - test_size))  # Сколько точек оставить

    indices = np.random.RandomState(random_state).choice(  # Случайный выбор точек
        len(X_non_sv), n_to_keep, replace=False
    )

    # Объединяем опорные вектора с оставшимися точками
    X_keep = pd.concat([X[is_sv], X_non_sv.iloc[indices]])
    y_keep = pd.concat([y[is_sv], y_non_sv.iloc[indices]])

    return X_keep, y_keep


def plot_svm(X, y, ax, title):
    model = SVC(kernel="linear", C=100000)
    model.fit(X, y)

    data_df_setosa = pd.DataFrame(
        X[y == "setosa"], columns=["sepal_length", "petal_length"]
    )
    data_df_versicolor = pd.DataFrame(
        X[y == "versicolor"], columns=["sepal_length", "petal_length"]
    )

    ax.scatter(
        data_df_setosa["sepal_length"],
        data_df_setosa["petal_length"],
        label="Setosa",
        alpha=0.6,
    )
    ax.scatter(
        data_df_versicolor["sepal_length"],
        data_df_versicolor["petal_length"],
        label="Versicolor",
        alpha=0.6,
    )

    ax.scatter(
        model.support_vectors_[:, 0],
        model.support_vectors_[:, 1],
        s=400,
        facecolors="none",
        edgecolors="black",
        label="Support Vectors",
    )

    x1_p = np.linspace(min(X["sepal_length"]), max(X["sepal_length"]), 100)
    x2_p = np.linspace(min(X["petal_length"]), max(X["petal_length"]), 100)
    X1_p, X2_p = np.meshgrid(x1_p, x2_p)

    X_p = pd.DataFrame(
        np.vstack([X1_p.ravel(), X2_p.ravel()]).T,
        columns=["sepal_length", "petal_length"],
    )
    y_p = model.predict(X_p)

    X_p["species"] = y_p
    X_p_setosa = X_p[X_p["species"] == "setosa"]
    X_p_versicolor = X_p[X_p["species"] == "versicolor"]

    ax.scatter(
        X_p_setosa["sepal_length"],
        X_p_setosa["petal_length"],
        alpha=0.1,
        color="blue",
    )
    ax.scatter(
        X_p_versicolor["sepal_length"],
        X_p_versicolor["petal_length"],
        alpha=0.1,
        color="orange",
    )

    ax.set_title(title)
    ax.legend()
    return model


model1 = plot_svm(X, y, axes[0], "Original Data")
original_support_vectors = model1.support_vectors_


X_train_75, y_train_75 = get_reduced_dataset(X, y, 0.25, original_support_vectors)
model2 = plot_svm(X_train_75, y_train_75, axes[1], "75%")

X_train_50, y_train_50 = get_reduced_dataset(X, y, 0.5, original_support_vectors)
model3 = plot_svm(X_train_50, y_train_50, axes[2], "50%")

X_train_25, y_train_25 = get_reduced_dataset(X, y, 0.75, original_support_vectors)
model4 = plot_svm(X_train_25, y_train_25, axes[3], "25%")

plt.tight_layout()
plt.show()

print("\nNumber of total points:")
print(f"Original data: {len(X)}")
print(f"75%: {len(X_train_75)}")
print(f"50%: {len(X_train_50)}")
print(f"25%: {len(X_train_25)}")
