# Метод опорных векторов(Support Vector Machines, SVM) - классификация и регрессия

# Разделяющая классификация
# Выбирается линия с наибольшим отступом

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# iris = sns.load_dataset("iris")
# print(iris.head())

# data = iris[["sepal_length", "petal_length", "species"]]
# data_df = data[(data["species"] == "setosa") | (data["species"] == "versicolor")]
# print(data.head())

# X = data_df[["sepal_length", "petal_length"]]
# y = data_df["species"]

# data_df_setosa = data_df[data_df["species"] == "setosa"]
# data_df_versicolor = data_df[data_df["species"] == "versicolor"]

# plt.scatter(data_df_setosa["sepal_length"], data_df_setosa["petal_length"])
# plt.scatter(data_df_versicolor["sepal_length"], data_df_versicolor["petal_length"])

# model = SVC(kernel="linear", C=100000)
# model.fit(X, y)

# print(model.support_vectors_[:, 0], model.support_vectors_[:, 1])

# plt.scatter(
#     model.support_vectors_[:, 0],
#     model.support_vectors_[:, 1],
#     s=400,
#     facecolors="none",
#     edgecolors="black",
# )


# x1_p = np.linspace(min(data_df["sepal_length"]), max((data_df["sepal_length"])), 100)
# x2_p = np.linspace(min(data_df["petal_length"]), max((data_df["petal_length"])), 100)
# X1_p, X2_p = np.meshgrid(x1_p, x2_p)

# X_p = pd.DataFrame(
#     np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=["sepal_length", "petal_length"]
# )
# y_p = model.predict(X_p)

# X_p["species"] = y_p

# X_p_setosa = X_p[X_p["species"] == "setosa"]
# X_p_versicolor = X_p[X_p["species"] == "versicolor"]

# plt.scatter(X_p_setosa["sepal_length"], X_p_setosa["petal_length"], alpha=0.1)
# plt.scatter(X_p_versicolor["sepal_length"], X_p_versicolor["petal_length"], alpha=0.1)

# plt.show()

# В случае если данные перекрываются то идеальной границы не существует. У модели существует гиперпараметры которые определяют "размытие" отступа

# iris = sns.load_dataset("iris")
# print(iris.head())
#
# data = iris[["sepal_length", "petal_length", "species"]]
# data_df = data[(data["species"] == "virginica") | (data["species"] == "versicolor")]
# print(data.head())
#
# X = data_df[["sepal_length", "petal_length"]]
# y = data_df["species"]
#
# data_df_virginica = data_df[data_df["species"] == "virginica"]
# data_df_versicolor = data_df[data_df["species"] == "versicolor"]
#
# c_value = [[10000,1000,100,10,1],[1,0.1,0.01,0.001,0.0001]]
#
# fig,ax = plt.subplots(2,5,sharex=True,sharey=True)
# for i in range (2):
#     for j in range(5):
#         ax[i][j].scatter(data_df_virginica["sepal_length"], data_df_virginica["petal_length"])
#         ax[i][j].scatter(data_df_versicolor["sepal_length"], data_df_versicolor["petal_length"])
# # Если с большое то отступ задается жестко, чем меньше с тем меньше отступ те более размытым
#         model = SVC(kernel="linear",C=c_value[i][j])
#         model.fit(X, y)
#         # print(model.support_vectors_[:, 0], model.support_vectors_[:, 1])
#
#         ax[i][j].scatter(
#             model.support_vectors_[:, 0],
#             model.support_vectors_[:, 1],
#             s=10,
#             facecolors="none",
#             edgecolors="black",
#         )
#
#         x1_p = np.linspace(min(data_df["sepal_length"]), max((data_df["sepal_length"])), 100)
#         x2_p = np.linspace(min(data_df["petal_length"]), max((data_df["petal_length"])), 100)
#         X1_p, X2_p = np.meshgrid(x1_p, x2_p)
#
#         X_p = pd.DataFrame(np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=["sepal_length", "petal_length"])
#         y_p = model.predict(X_p)
#
#         X_p["species"] = y_p
#
#         X_p_virginica = X_p[X_p["species"] == "virginica"]
#         X_p_versicolor = X_p[X_p["species"] == "versicolor"]
#
#         ax[i][j].scatter(X_p_virginica["sepal_length"], X_p_virginica["petal_length"], alpha=0.01)
#         ax[i][j].scatter(X_p_versicolor["sepal_length"], X_p_versicolor["petal_length"], alpha=0.01)
#
# plt.show()
#

# Достоинства
# - Зависимость от небольшого числа опорных векторов => компактность модели
# - После обучения модель работает быстро
# - На работу метода вличют ТОЛЬКО точки, находящиеся возле отступов, поэтому методы подходят для многомерных данных

# Недостатки
# - при большом количестве обучающих образцов могут быть значительные вычислительные затраты
# - Большая зависимость от размытия отступа. Поиск отступа может занять много времени
# - У результатов отсутствует вероятнастная интерпретация


## Деревья решений и случайные леса
# СЛ - непараметрический алгоритм

# СЛ - пример ансамблиевого метода основанного на агрегации результатов нескольких простых моделей
# В реализации дерева принятия реений в машинном обучении вопросы обычно ведут к разделению данных по осям
# те каждыйы узел разбивает данные на две группы по одному из признаков
# iris = sns.load_dataset("iris")
# species_int = []
# for r in iris.values:
#     if r[4] == "setosa":
#         species_int.append(1)
#     elif r[4] == "versicolor":
#         species_int.append(2)
#     else:
#         species_int.append(3)

# species_int_df = pd.DataFrame(species_int)
# print(iris.head())

# data = iris[["sepal_length", "petal_length"]]
# data["species"] = species_int
# data_df = data[(data["species"] == 3) | (data["species"] == 2)]
# print(data.head())

# X = data_df[["sepal_length", "petal_length"]]
# y = data_df["species"]

# data_df_setosa = data_df[data_df["species"] == 3]
# data_df_versicolor = data_df[data_df["species"] == 2]

# plt.scatter(data_df_setosa["sepal_length"], data_df_setosa["petal_length"])
# plt.scatter(data_df_versicolor["sepal_length"], data_df_versicolor["petal_length"])

# model = DecisionTreeClassifier(max_depth=3)
# model.fit(X, y)

# x1_p = np.linspace(min(data_df["sepal_length"]), max((data_df["sepal_length"])), 100)
# x2_p = np.linspace(min(data_df["petal_length"]), max((data_df["petal_length"])), 100)

# X1_p, X2_p = np.meshgrid(x1_p, x2_p)

# X_p = pd.DataFrame(np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=["sepal_length", "petal_length"])

# y_p = model.predict(X_p)

# X_p["species"] = y_p

# X_p_setosa = X_p[X_p["species"] == "virginica"]
# X_p_versicolor = X_p[X_p["species"] == "versicolor"]

# # plt.scatter(X_p_setosa["sepal_length"], X_p_setosa["petal_length"], alpha=0.05)
# # plt.scatter(X_p_versicolor["sepal_length"], X_p_versicolor["petal_length"], alpha=0.05)

# plt.contourf(X1_p, X2_p, y_p.reshape(X1_p.shape), alpha=0.3, levels=2, cmap="rainbow", zorder=1)

# plt.show()
