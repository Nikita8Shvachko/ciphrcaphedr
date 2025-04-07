# Линейная регрессия

import matplotlib.pyplot as plt
import numpy as np

# Задача: на основе наблюдаемых точек построить прямую, которая отображает связь между двумя или более переменными.
# Регрессия пытается "подогнать" функцию к наблюдаемым данным, чтобы спрогнозировать новые данные.
# Линейная регрессия подгоняет данные к прямой линии, пытаемся установить линейную связь между переменными и предсказать
# новые данные.


# features, target = make_regression(
#     n_samples=100,
#     n_features=1,
#     n_informative=1,
#     n_targets=1,
#     noise=10,
#     random_state=1
# )
# print(features.shape)
#
# print(target.shape)
#
# model = LinearRegression().fit(features, target)
#
# plt.scatter(features, target)
#
# x = np.linspace(features.min(), features.max(), 100)
# # y = kx +b
# plt.plot(x, model.coef_[0] * x + model.intercept_,color='red')
#
# plt.show()


# Простая линейная регрессия: исследование линейных зависимостей

# Преимущества:
# + Возможность прогнозирования на новых данных.
# + Анализ взаимного влияния переменных.

# Недостатки:
# - Обучаемые данные не всегда точно лежат на прямой из-за шума, что приводит к погрешностям.
# - Невозможность прогнозирования вне диапазона имеющихся данных.

# Важно: Данные для модели должны быть репрезентативной выборкой из совокупности.

# Определение остатков: это расстояния между точками данных и ближайшими точками на прямой (по вертикали).
# Цель: минимизация суммы квадратов остатков.
# Обучение модели сводится к минимизации функции потерь, где квадраты остатков рассматриваются как площади квадратов.

# Решение: - численное - проще, но не всегда точно;
#          - аналитическое - сложнее, но точно(опираются на математическую теорию).

# y = w_0 +w_1 * x

# данные имеют вид: (x_i, y_i)...
# n - число точек

# w_1 = ...
#
#
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
x = data[:, 0]
y = data[:, 1]
# print(x)
# print(y)
# n = len(x)
# w_1 = (n * sum(x[i] * y[i] for i in range(n)) - (sum(x[i] for i in range(n))) * (sum(y[i] for i in range(n)))) / (
#         n * (sum(x[i] ** 2 for i in range(n))) - (sum(x[i] for i in range(n))) ** 2)
#
# w_0 = (sum(y[i] for i in range(n)) - w_1 * (sum(x[i] for i in range(n)))) / n
#
# print(w_0, w_1)
#
# plt.scatter(x, y)
# plt.plot(x, w_0 + w_1 * x, color='red')

# Другой метод решения: метод обратных матриц.
# y = w_1*x + w_0       w = [ w_1; w_0 ]^T
#
# x_1 = np.vstack([x, np.ones(len(x))]).T
# # print(x_1)
# w = inv(x_1.T.dot(x_1)) @ x_1.T @ y
# print(w)
#
# plt.scatter(x, y)
# # plt.plot(x, w[0] * x + w[1], color='green')
# plt.show()

# 3 Способ решения: QR разложение X = Q * R -> w = R^-1 * Q^T * y
# QR разложение позволяет минимизировать ошибку от вычисления обратной матрицы.
#
# Q,R = np.linalg.qr(x_1)
# w_qr = inv(R).dot(Q.T).dot(y)
# print(w_qr)

# 4 Способ решения: Градиентный спуск
# Метод оптимизации позволяет определить угловой коэффициент и изменить его, чтобы мин/макс функцию потерь.
# Для больших угловых коэффициентов

# def f(x):
#     return (x-3)**2 +4
#
# def df(x):
#     return 2*(x-3)

# x = np.linspace(-10, 10, 100)
#
# ax = plt.gca()
# ax.xaxis.set_major_locator(plt.MultipleLocator(1))
#
# # plt.plot(x, f(x))
# plt.plot(x, df(x))
#
# # plt.show()
#
#
#
#
#
# data = np.array(
#     [
#         [1, 5],
#         [2, 7],
#         [3, 7],
#         [4, 10],
#         [5, 11],
#         [6, 14],
#         [7, 17],
#         [8, 19],
#         [9, 22],
#         [10, 28]
#     ]
# )
# x = data[:, 0]
# y = data[:, 1]
#
# n = len(x)
#
# w1 = 0.0
# w0 = 0.0
# L = 0.001
#
# iterations = 100_000
#
# for i in range(iterations):
#     d_w0 = 2*sum(y[i]+w0-w1*x[i] for i in range(n))
#     d_w1 = 2*sum(x[i]*(-y[i]-w0+w1*x[i]) for i in range(n))
#
#     w1 -= L * d_w1
#     w0 -= L * d_w0
#
#
# print(w1, w0)


w1 = np.linspace(-10, 10, 100)
w0 = np.linspace(-10, 10, 100)


def E(w1, w0, x, y):
    return sum(y[i] - (w0 + w1 * x[i]) ** 2 for i in range(len(x)))


W1, W0 = np.meshgrid(w1, w0)

EW = E(W1, W0, x, y)

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection='3d' )
ax.view_init(175, 70)
w1_fit = 2.4
w0_fit = 0.8

E_fit = E(w1_fit, w0_fit, x, y)
ax.scatter3D(w1_fit, w0_fit, E_fit, color='red')

ax.plot_surface(W1, W0, EW)
# angle = 45


plt.show()
