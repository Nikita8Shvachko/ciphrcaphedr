import numpy as np

# Суммирование значений и другие агрегатные функции
#
# rng = np.random.default_rng(1)
# s  = rng.random(50)
#
# print(s)
# print(sum(s))
# print(np.sum(s))
#
# a = np.array([
#     [1, 2, 3,4,5],
#     [6, 7, 8,9,10],
# ])
#
#
# print(np.sum(a))
# print(np.sum(a, axis=0))
# print(np.sum(a, axis=1))
#
# print(a.min())
# print(a.min(0))
# print(a.min(1))
#
# # Nan = not a number
# print(np.nanmin(a))
# print(np.nanmin(a, axis=0))
# print(np.nanmin(a, axis=1))

# Broadcasting
# Set of rules which allows to perform binary operations on arrays with different shapes
#
# a = np.array([0, 1, 2])
# b = np.array([5, 5, 5])
# print(a + b)
# print(a + 5)  # 5 is broadcasted to each element ie [5, 5, 5]
#
# a = np.array([[0, 1, 2], [3, 4, 5]])
# print(a + 5)
# a = np.array([0,1,2])
# b = np.array([[0],[1],[2]])
#
# print(a + b)

# Broadcasting rules
# 1. If dimension mismatch, the smaller array is broadcasted to match the larger array
# 2. If shapes don't match,but ones in any dimension is equal to 1 , the smaller array is broadcasted to match the larger array
# 3. If in any dimension the sizes don't match and none is equal to 1, an error is raised
# a = np.array([[0, 1, 2], [3, 4, 5]])
# b = np.array([5, 5, 5])
# print(a + b)
# print(a.ndim, a.shape)
# print(b.ndim, b.shape)
# a 2 (2, 3)
# b 1 (3,)


# a = np.ones((2, 3))
# b = np.arange(3)
# print(a)
# print(b)
# print(a.ndim, a.shape)
# print(b.ndim, b.shape)
# # a 2 (2, 3) -> (2,3) -> (2,3)
# # b 1   (3,) -> (1,3) -> (2,3)
#
# c = a + b
# print(c, c.ndim, c.shape) # c 2 (2, 3)

# a = np.arange(3).reshape((3, 1))
# b = np.arange(3)
# print(a)
# print(b)
# print(a.ndim, a.shape)
# print(b.ndim, b.shape)
# # a 2 (3, 1) -> (3,1) -> (3,3)
# # b 1 (3,)  ->  (1,3) -> (3,3)
# c = a + b
# # [0 0 0]   [0 1 2]
# # [1 1 1] + [0 1 2]
# # [2 2 2]   [0 1 2]
# print(c, c.ndim, c.shape)

# a = np.ones((3,2))
# b = np.arange(3)
# # a 2 (3, 2) -> (3,2) -> (3,2)
# # b 1   (3,) -> (1,3) -> (3,3)
#
# c = a+b


## todo: 1
## 1. Что надо изменить в последнем примере, чтобы он заработал без ошибок (транслирование)?
# X = np.array([
#     [1,2,3,4,5,6,7,8,9],
#     [9,8,7,6,5,4,3,2,1]
# ])
#
# Xmean0 = X.mean(0)
# print(Xmean0)
# Xcenter0 = X - Xmean0
# print(Xcenter0)
#
# Xmean1 = X.mean(1)
# print(Xmean1)
# Xmean1 = Xmean1[:, np.newaxis]
# Xcenter1 = X - Xmean1
# print(Xcenter1)

# x = np.linspace(0, 5, 50)
# y = np.linspace(0, 5, 50)[:, np.newaxis]
#
# z = np.sin(x) ** 3 + np.cos(20 + y * x) * np.sin(y)
# print(z.shape)
# import matplotlib.pyplot as plt
#
# plt.imshow(z)
# plt.colorbar()
# plt.show()

# x = np.array([[1, 2, 3, 4, 5])
# y = np.array([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
# print(x < 3)
# print(np.less(x, 3))
#
# print(np.sum(x < 3, axis=0))  # Number of elements <3
#
# ## Todo: 2
## 2. Пример для y. Вычислить количество элементов (по обоим размерностям),
# значения которых больше 3 и меньше 9
y = np.array([[1, 2, 3, 4, 5],
              [6, 7, 8, 9, 10]])
n = y[(3 < y) & (y < 9)]
print(n)

# # Masks - boolean arrays
# x = np.array([1, 2, 3, 4, 5])
# y = print(x < 3)
# print(x[x<3])
# print(bin(42))
# print(bin(59))
# print(bin(42 &59))
#

# Index Vectorisation

# x = np.array([0,1, 2, 3, 4, 5, 6, 7, 8, 9])
# index = [1,5,7]
# print(x[index])
# index = [[1,5,7],[2,4,8]]
# print(x[index])
# Result is show shape of index array and not an original array


# x = np.arange(12).reshape((3, 4))
# print(x)
# print(x[2])
# print(x[2, [2, 0, 1]])
# print(x[1, [2, 0, 1]])


# x = np.arange(10)
# i = np.array([2, 1, 8, 4])
# print(x)
# x[i]=999
# print(x)


# Sorting
# x =  [3,2,3,5,6,7,3,6,3,2]
# print(sorted(x))
# print(np.sort(x)) # faster  on large data

# ## Structured arrays
# data = np.zeros(4, dtype={
#     'names': (
#         'name', 'age'
#     ),
#     'formats': (
#         'U10', 'i4'
#     )
# })
#
# print(data.dtype)
#
# names = ['name1', 'name2', 'name3', 'name4']
# ages = [10, 20, 30, 40]
# data['name'] = names
# data['age'] = ages
# print(data)
# print(data[data['age'] < 25]['name'])
#
# # Array of appointments
#
# data_rec = data.view(np.recarray)
# print(data_rec)
# print(data_rec[0])
# print(data_rec[-1].name)
