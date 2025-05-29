# Переобучение присуще всем деревьям принятия решений

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

# iris = sns.load_dataset("iris")
# # print(iris.head())
# # sns.pairplot(iris, hue="species")


# species_int = []
# for r in iris.values:
#     if r[4] == "setosa":
#         species_int.append(1)
#     elif r[4] == "versicolor":
#         species_int.append(2)
#     else:
#         species_int.append(3)

# species_int_df = pd.DataFrame(species_int)

# data = iris[["sepal_length","petal_length"]]
# data["species"]= species_int

# print(data.head())
# print(data.shape)

# data_versicolor = data[data["species"] == 2]
# data_virginica = data[data["species"] == 3]

# print(data_versicolor.shape)
# print(data_virginica.shape)

# data_versicolor_A = data_versicolor.iloc[:25,:]
# data_versicolor_B = data_versicolor.iloc[25:,:]

# data_virginica_A = data_virginica.iloc[:25,:]
# data_virginica_B = data_virginica.iloc[25:,:]

# print(data_versicolor_A.shape)
# print(data_versicolor_B.shape)

# print(data_virginica_A.shape)
# print(data_virginica_B.shape)

# data_df_A = pd.concat([data_versicolor_A, data_virginica_A], ignore_index=True)
# data_df_B = pd.concat([data_versicolor_B, data_virginica_B], ignore_index=True)


# fig,ax = plt.subplots(2,4)

# max_depth = [1,3,5,7]
# X  = data_df_A[["sepal_length","petal_length"]]

# x1_p = np.linspace(min(data["sepal_length"]),max(data["sepal_length"]),100)
# x2_p = np.linspace(min(data["petal_length"]),max(data["petal_length"]),100)

# X1_p , X2_p = np.meshgrid(x1_p,x2_p)

# x1_p,x2_p = np.meshgrid(x1_p,x2_p)

# X_p = pd.DataFrame(np.vstack([X1_p.ravel(),X2_p.ravel()]).T,columns=["sepal_length","petal_length"])


# y = data_df_A["species"]

# j=0
# for md in max_depth:
#     model = DecisionTreeClassifier(max_depth=md)
#     model.fit(X,y)
#     ax[0][j].scatter(data_virginica_A["sepal_length"],data_virginica_A["petal_length"])
#     ax[0][j].scatter(data_versicolor_A["sepal_length"],data_versicolor_A["petal_length"])

#     y_p = model.predict(X_p)
#     ax[0][j].contourf(X1_p,X2_p,y_p.reshape(X1_p.shape),alpha=0.2,cmap="RdBu")
#     j+=1


# X  = data_df_B[["sepal_length","petal_length"]]

# X_p = pd.DataFrame(np.vstack([X1_p.ravel(),X2_p.ravel()]).T,columns=["sepal_length","petal_length"])

# y = data_df_B["species"]

# j=0
# for md in max_depth:
#     model = DecisionTreeClassifier(max_depth=md)
#     model.fit(X,y)
#     ax[1][j].scatter(data_virginica_B["sepal_length"],data_virginica_B["petal_length"])
#     ax[1][j].scatter(data_versicolor_B["sepal_length"],data_versicolor_B["petal_length"])

#     y_p = model.predict(X_p)
#     ax[1][j].contourf(X1_p,X2_p,y_p.reshape(X1_p.shape),alpha=0.2,cmap="RdBu")
#     j+=1


# plt.show()


# Ансамблиевые методы В основе лежит идея о том, что несколько слабых моделей могут дать лучший результат, чем одна сильная модель. тем самым уменьшается переобучение.

# Баггинг усредняет результаты -> оптимальной классификации
# Ансамбль случайных деревьев называется случайным лесом


# iris = sns.load_dataset("iris")
# # print(iris.head())
# # sns.pairplot(iris, hue="species")


# species_int = []
# for r in iris.values:
#     if r[4] == "setosa":
#         species_int.append(1)
#     elif r[4] == "versicolor":
#         species_int.append(2)
#     else:
#         species_int.append(3)

# species_int_df = pd.DataFrame(species_int)

# data = iris[["sepal_length","petal_length"]]
# data["species"]= species_int


# data_setosa = data[data["species"] == 1]
# data_versicolor = data[data["species"] == 2]
# data_virginica = data[data["species"] == 3]


# data_versicolor_A = data_versicolor.iloc[:25,:]
# data_versicolor_B = data_versicolor.iloc[25:,:]

# data_virginica_A = data_virginica.iloc[:25,:]
# data_virginica_B = data_virginica.iloc[25:,:]

# data_setosa_A = data_setosa.iloc[:25,:]
# data_setosa_B = data_setosa.iloc[25:,:]


# data_df_A = pd.concat([data_versicolor_A, data_virginica_A, data_setosa_A], ignore_index=True)
# data_df_B = pd.concat([data_versicolor_B, data_virginica_B, data_setosa_B], ignore_index=True)


# fig,ax = plt.subplots(1,3,sharex=True,sharey=True)

# # max_depth = [1,3,5,7]


# x1_p = np.linspace(min(data["sepal_length"]),max(data["sepal_length"]),100)
# x2_p = np.linspace(min(data["petal_length"]),max(data["petal_length"]),100)

# X1_p , X2_p = np.meshgrid(x1_p,x2_p)

# x1_p,x2_p = np.meshgrid(x1_p,x2_p)

# X_p = pd.DataFrame(np.vstack([X1_p.ravel(),X2_p.ravel()]).T,columns=["sepal_length","petal_length"])


# X  = data_df_A[["sepal_length","petal_length"]]
# y = data_df_A["species"]

# md = 6
# model1 = DecisionTreeClassifier(max_depth=md)
# model1.fit(X,y)
# y_p1 = model1.predict(X_p)
# ax[0].scatter(data_setosa_A["sepal_length"],data_setosa_A["petal_length"])
# ax[0].scatter(data_virginica_A["sepal_length"],data_virginica_A["petal_length"])
# ax[0].scatter(data_versicolor_A["sepal_length"],data_versicolor_A["petal_length"])
# ax[0].contourf(X1_p,X2_p,y_p1.reshape(X1_p.shape),alpha=0.2,cmap="rainbow",levels=2)

# # Bagging
# from sklearn.ensemble import BaggingClassifier
# model2 = DecisionTreeClassifier(max_depth=md)
# b = BaggingClassifier(model2,n_estimators=20,max_samples=0.5)
# b.fit(X,y)


# y_p2 = b.predict(X_p)
# ax[1].scatter(data_setosa_A["sepal_length"],data_setosa_A["petal_length"])
# ax[1].scatter(data_virginica_A["sepal_length"],data_virginica_A["petal_length"])
# ax[1].scatter(data_versicolor_A["sepal_length"],data_versicolor_A["petal_length"])
# ax[1].contourf(X1_p,X2_p,y_p2.reshape(X1_p.shape),alpha=0.2,cmap="rainbow",levels=2)


# #  Random Forest


# rf = RandomForestClassifier(n_estimators=20,max_depth=md)
# rf.fit(X,y)
# y_p3 = rf.predict(X_p)
# ax[2].scatter(data_setosa_A["sepal_length"],data_setosa_A["petal_length"])
# ax[2].scatter(data_virginica_A["sepal_length"],data_virginica_A["petal_length"])
# ax[2].scatter(data_versicolor_A["sepal_length"],data_versicolor_A["petal_length"])
# ax[2].contourf(X1_p,X2_p,y_p3.reshape(X1_p.shape),alpha=0.2,cmap="rainbow",levels=2)


# plt.show()


# Достоинства
# - Простота и быстрота Распараллеливание процесса обучения - выигрыш во времени
# - Вероятноствая классификация
# - Модель непараметрическая, не требуется выбирать параметры
# Недостатки
# - Сложность интерпретации


# Метод главных компонент(PCA - Principal Component Analysis) - алгоритм обучения без учителя
# Часто используют для понижения размерности

# Задача машинного обучения без учителя состоит в выяснении зависимости между признаками
# В PCA выполняется качественная оценка этой зависимости путем поиска главных осей координат и их использования для описания наборов данных


# пример


iris = sns.load_dataset("iris")


species_int = []
for r in iris.values:
    if r[4] == "setosa":
        species_int.append(1)
    elif r[4] == "versicolor":
        species_int.append(2)
    else:
        species_int.append(3)


data = iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
data["species"] = species_int


data_setosa = data[data["species"] == 1]
data_versicolor = data[data["species"] == 2]
data_virginica = data[data["species"] == 3]

data_setosa_A = data_setosa.iloc[:25, :]
data_setosa_B = data_setosa.iloc[25:, :]

data_versicolor_A = data_versicolor.iloc[:25, :]
data_versicolor_B = data_versicolor.iloc[25:, :]

data_virginica_A = data_virginica.iloc[:25, :]
data_virginica_B = data_virginica.iloc[25:, :]

data_df_A = pd.concat(
    [data_setosa_A, data_versicolor_A, data_virginica_A], ignore_index=True
)
data_df_B = pd.concat(
    [data_setosa_B, data_versicolor_B, data_virginica_B], ignore_index=True
)


pca = PCA(n_components=2)
X = data_df_A[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
X_pca = pca.fit_transform(X)


pca_df = pd.DataFrame(data=X_pca, columns=["PC1", "PC2"])
pca_df["species"] = data_df_A["species"]


plt.figure(figsize=(10, 6))
plt.scatter(
    pca_df[pca_df["species"] == 1]["PC1"],
    pca_df[pca_df["species"] == 1]["PC2"],
    label="Setosa",
)
plt.scatter(
    pca_df[pca_df["species"] == 2]["PC1"],
    pca_df[pca_df["species"] == 2]["PC2"],
    label="Versicolor",
)
plt.scatter(
    pca_df[pca_df["species"] == 3]["PC1"],
    pca_df[pca_df["species"] == 3]["PC2"],
    label="Virginica",
)


plt.show()

print(pca.components_)
print(pca.explained_variance_ratio_)
print(pca.mean_)


# + простота интерпретации, эффективность в работе с многомерными данными
# - аномальные значения могут сильно влиять на результат
