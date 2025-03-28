# Введение в машиное обучение Machine Learning - ML
import matplotlib.pyplot as plt
# Задача машинного обучения:
# Требуется подогнать заданный набор точек данными под соответствующую функцию(отображение входа на выход),
# которая улавливает важные сигналы в данных и игнорирует помехи, а затем убедиться, что модель работает хорошо на
# новых данных.


# Типы машинного обучения:

# 1) Обучение с учителем - SML (Supervised learning)
#     Обучение с учителем моделирует отношение между признаками и метками. Такие модели могут быть использованы для предсказания
#     меток новых объектов на основе обучающих данных(маркированных). После построения модели можно использовать ее для
#     классификации новых объектов ранее не виденных моделью.

# - задачи классификации (метки - дискретные: два объекта и более)
# - задачи регрессии (метки/результат - непрерывные: один объект)


# 2) Обучение без учителя - UML (Unsupervised learning)
# - задачи кластеризации (выделяет отдельные группы данных(кластеры))
# - понижение размерности (поиск более сжатого представления данных, чтобы улучшить качество модели )


# 3) Частичное обучение - SSL (Semi supervised learning) - не все данные промаркированы.

# 4) Метод обучения с подкреплением - RL (Reinforcement learning) - Система обучения улучшает свои характеристики на
# основе взаимодействия (обратной связи) со средой. При этом взаимодействии система получает сигналы (функции наград),
# которые несут в себе информацию насколько хорошо/плохо система решила задачу(с точки зрения системы).
# Итоговая награда максимальная.


# пример данных для обучения

# import seaborn as sns
#
# iris = sns.load_dataset('iris')
#
# print(iris.head())

# Строки - отдельные объекты - образцы (samples)
# Столбцы - признаки (features) - соответствуют конкретным наблюдениям (observations)
# Матрицы признаков (features matrix) - матрица, в которой строки X - соответствуют объектам, а столбцы Y - признакам

# Целевой массив, массив меток (targets) - вектор соответствующих объектов меток [1 x число образцов] - данные, которые
# мы хотим предсказать на основе имеющихся признаков.

# Зависимые(метка) и независимые переменные (признаки)


# Процесс построения модели машинного обучения:


# 1 Предварительная обработка данных
#     - На вход поступают необработанные данные и метки
#     - Происходит выбор признаков, масштабирование признаков (нормализация, стандартизация)
#     - Понижение размерности
#     - Выборка образцов (семплирование)
#     - Разделение данных на обучающую и тестовую выборки


# 2 Обучение модели
#     - Выбор модели
#     - Перекрестная проверка (кросс-валидация)
#     - Метрики эффективности модели
#     - Подбор и оптимизация гиперпараметров модели. Параметры,
#         которые получаются не из данных, а являются характеристиками модели.


# 3 Оценка и формирование финальной модели

# 4 Предсказание(прогнозирование, использование модели)


# Наш стэк для изучения ML пока...
# Scikit-learn
# 1 Выбираем класс модели
# 2 Устанавливаем гиперпараметры модели
# 3 Создаем матрицу признаков и целевой массив
# 4 Обучение модели fit()
# 5 Применим модель к новым данным
#     - predict() ( с учителем)
#     - predict() или transform() (без учителя)

# Обучение с учителем - линейная регрессия

## Простая линейная регрессия - уравнение вида
#
# y = a + b * x

import numpy as np

np.random.seed(0)
x = 10 * np.random.rand(100)

y = 2 * x + np.random.randn(100)

# Теперь научим модель на наших данных простой линейной регрессией

# 1 - Выбираем класс модели
from sklearn.linear_model import LinearRegression

# 2 - Устанавливаем гиперпараметры модели
model = LinearRegression()  # линейная регрессия(аргументы: fit_intercept=True(Это параметр отвечает за наличие смещения)
# 3 - Создаем матрицу признаков и целевой массив
model.fit(x[:, np.newaxis], y)
# 4 - Обучение модели на наших данных
y_pred = model.predict(x[:, np.newaxis])
# 5 - Применим модель к новым данным - прогнозирование
plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
plt.show()

# Коэффициенты модели которые она нашла при обучении на наших данных
print(model.coef_[0])
print(model.intercept_)

# Наши изначальные коэффициенты
print(2, 0)
