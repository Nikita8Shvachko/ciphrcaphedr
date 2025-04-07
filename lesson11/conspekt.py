import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Переобучение и дисперсия

# Линейная регрессия
# Цель не в минимизации суммы квадратов а в том чтобы делать правильные предсказания на новых данных = минимизировать дисперсию

# Переобученные модели - очень чувствительные к выбросам = высокая дисперсия

# Пожтому к моделям специально добавляется смешение
#
# Смещение модели означает что предпочтение отдается определноой схеме (например прямая линия ) а не чемото со  сложной структурой

# Если в модель добавить смещения то модель может получиться недообученной

# Поэтому надо балансировать!

# Есть два вида регрессии
# - Гребневая регрессия (ridge) - добавляет смещение в виде штрафа изза этого хуже идет подгонка

# - Лассо регрессия - удаление некоторых переменных


# Механически применять линейную регрессию к данным , сделать на основе полученной модели какойто прогноз и думать что все в порядке нельзя


data = np.array(
    [
        [1, 5],
        [2, 7],
        [3, 7],
        [4, 10],
        [5, 11],
        [6, 14],
        [7, 17],
        [8, 19],
        [9, 22],
        [10, 28]
    ]
)

# Градиентный спуск - пакетный градиентный спуск. Для работы используюстя ВСЕ доступные обучающие данные
# Стохастический градиентный спуск, на каждой итерации обучаемся только по одной выборке из данных
# - сокращение вычислений
# - вносим смещение => боремся с переобучением
# Мини-пакетный градиентный спуск на каждой итерации используется несколько выборок
# x = data[:, 0]
# y = data[:, 1]
#
# n = len(x)
#
# w1 = 0.0
# w0 = 0.0
# L = 0.001
#
# iterations = 100_00

# Sample size - размер выборок
#
# sample_size = 1
#
# for i in range(iterations):
#     idx = np.random.choice(n, sample_size,replace=False)
#     d_w0 = 2 * sum(-y[idx] + w0 + w1 * x[idx] )
#     d_w1 = 2 * sum((x[idx] * (-y[idx] + w0 + w1 * x[idx])) )
#
#     w1 -= L * d_w1
#     w0 -= L * d_w0
#
# print(w1, w0)
# # Как оценить насколько сильно промахиваются прогнозы при использовании линейной регрессии
#
data_df = pd.DataFrame(data)
# print(data_df.corr(method='pearson'))


# Обучающие и тестовые выборки
# Основной метод борьбы с переобучением - набор данных делится на обучающий и тестовый выборку
#
# Во всех видах машинного обучения с учителем есть обучающий и тестовый выборки
#
# Обычная пропорция - 2/3 обучающий, 1/3 тестовый(4/5 к тестовому, 1/5 к обучающему и тд )
#
# X = data_df.values[:, :-1]
# y = data_df.values[:, -1]
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3)
#
# print(X_train)
# print(y_train)
#
# print(X_test)
# print(y_test)
#
# model = LinearRegression()
# model.fit(X_train, y_train)
# print(model.score(X_test, y_test))
#
# Коэфициент детерминации - коэффициент определяющий меру детерминации модели что означает .(чем ближе к 1 тем лучше модель)


kfold = KFold(n_splits=3, random_state=1, shuffle=True)

X = data_df.values[:, :-1]
y = data_df.values[:, -1]

model = LinearRegression()
results = cross_val_score(model, X, y, cv=kfold)
print(results)
print(results.mean(), results.std())
# Метрики показываеют насколько единообразно ведет себя модель на разных выборках
# Возможно использование поэлементной перекрестной валидации - мало данных
# Случайная валидация


# Валидационная выборка - для сравнения различных моделей и конфигураций


data_df = pd.read_csv('multiple_independent_variable_linear.csv')
print(data_df.head())

X = data_df.values[:, :-1]
Y = data_df.values[:, -1]

model = LinearRegression().fit(X, Y)
print(model.coef_, model.intercept_)
x1 = X[:, 0]
x2 = X[:, 1]
y = Y

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x1, x2, y)

x1_ = np.linspace(x1.min(), x1.max(), 100)
x2_ = np.linspace(x2.min(), x2.max(), 100)
x1_, x2_ = np.meshgrid(x1_, x2_)
y_ = model.coef_[0] * x1_ + model.coef_[1] * x2_ + model.intercept_
ax.plot_surface(x1_, x2_, y_, cmap='gray', alpha=0.3)
plt.show()
