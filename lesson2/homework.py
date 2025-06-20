import numpy as np

## 1. Что надо изменить в последнем примере, чтобы он заработал без ошибок (транслирование)?
a = np.ones((3, 2))
print("\n", a)
b = np.arange(3)
print("\n", b)
# a 2 (3, 2) -> (3,2) -> (3,2)
# b 1   (3,) -> (1,3) -> (3,3)

c = a + b.reshape(3, 1)
print("\n", c)


## 2. Пример для y. Вычислить количество элементов (по обоим размерностям),
# значения которых больше 3 и меньше 9
# y = np.array([[1, 2, 3, 4, 5],
#               [6, 7, 8, 9, 10]])
#
# print(np.sum((y<3) & (y < 9),axis=0))
# print(np.sum((y<3) & (y < 9),axis=1))
# print(np.sum((y<3) & (y < 9)))
