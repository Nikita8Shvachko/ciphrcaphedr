import numpy as np
## 1. Какие еще существуют коды типов?
# print(np.typecodes.keys())

## 2. Напишите код, подобный приведенному выше, но с другим типом
# a = np.array([1.22, 2.33, 3.44], dtype=np.float32)
# print(type(a), a)

# x = list[1, 2, 3]
# print(type(x))
#
# x = {'name': 'John', 'age': 30}
# print(type(x))
#
# x = (1, 2, 3)
# print(type(x))

# a1 = array.array('i', [1, 2, 3])
# print(type(a1))
# print(sys.getsizeof(a1))

## 3. Напишите код для создания массива с 5 значениями, располагающимися через равные интервалы в диапазоне от 0 до 1
# arr = np.linspace(0, 1, 5)
# print(arr)

## 4. Напишите код для создания массива с 5 равномерно распределенными случайными значениями в диапазоне от 0 до 1

# arr = np.random.uniform(0, 1, 5)
# print(arr)

## 5. Напишите код для создания массива с 5 нормально распределенными случайными значениями с мат. ожиданием = 0 и дисперсией 1
# arr = np.random.normal(0, 1, 5)
# print(arr)
## 6. Напишите код для создания массива с 5 случайнвми целыми числами в от [0, 10)
# arr = np.random.randint(0, 10, 5)
# print(arr)

## 7. Написать код для создания срезов массива 3 на 4

# arr = np.arange(12).reshape(3, 4)
# print( arr)
## - первые две строки и три столбца
# print(arr[:2, :3])

## - первые три строки и второй столбец
# print(arr[:, 1:2])

## - все строки и столбцы в обратном порядке
# print(arr[::-1, ::-1])

## - второй столбец
# print(arr[:, 1])

## - третья строка
# print(arr[2, :])


## 8. Продемонстрируйте, как сделать срез-копию
# arr = np.arange(12).reshape(3, 4)
# print(arr)
# copy = arr[:2, :3].copy()
# print(copy)

## 9. Продемонстрируйте использование newaxis для получения вектора-столбца и вектора-строки
# vector = np.array([1, 2, 3])
# print(vector)
# print(vector[:, np.newaxis])
# print( vector[np.newaxis, :])

## 10. Разберитесь, как работает метод dstack
# a = np.array([1, 2])
# b = np.array([3, 4])
# print(a, "\n", b, "\n")
# stacked = np.dstack((a, b))
# print(stacked)
## 11. Разберитесь, как работают методы split, vsplit, hsplit, dsplit
# array_split = np.arange(16).reshape(4, 4)
# splits = np.split(array_split, 2)  # split
# print("split:", splits)
#
# vsplits = np.vsplit(array_split, 2)  # vsplit
# print("vsplit:", vsplits)
#
# hsplits = np.hsplit(array_split, 2)  # hsplit
# print("hsplit:", hsplits)
#
# array_3d = np.arange(24).reshape(2, 3, 4)
# dsplits = np.dsplit(array_3d, 2)  # dsplit
# print("dsplit:", dsplits)

## 12. Привести пример использования всех универсальных функций, которые я привел
# arr_ufunc = np.array([1.1, 0.1, 1, 2, 3],dtype=np.float32)
# print("abs:", np.abs(arr_ufunc))  # abs
# print("sin:", np.sin(arr_ufunc))  # sin
# print("exp:", np.exp(arr_ufunc))  # exp
# print("sqrt:", np.sqrt(arr_ufunc + 1))  # sqrt
# print("log:", np.log(arr_ufunc + 1))  # log
