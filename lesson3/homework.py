# %%

import numpy as np
import pandas as pd

# 1. Привести различные способы создания объектов типа Series
# Для создания Series можно использовать
# - списки Python или массивы NumPy
# - скалярные значение
# - словари

ex1 = list([True, "Hello world", 3, 'a', 4.123])
ex2 = np.array([1, 2, 3, 4, 5])
ex3 = [1, 2, 3, 4, 5]

seriesex1 = pd.Series(ex1, index=['a', 'b', 'c', 'd', 'e'])
seriesex2 = pd.Series(ex2, index=['aa', 'bb', 'cc', 'dd', 'ee'])
seriesex3 = pd.Series(ex3)
seriesex4 = pd.Series(dict(zip(['a', 'b', 'c', 'd', 'e'], ['a', 22, '33', 443, 'hello'])))

print(seriesex1, seriesex2, seriesex3, seriesex4, sep='\n')

# %% 2. Привести различные способы создания объектов типа DataFrame.
# DataFrame. Способы создания
# - через объекты Series
# - списки словарей
# - словари объектов Series
# - двумерный массив NumPy
# - структурированный массив Numpy

df_ex1 = pd.DataFrame({'First': seriesex1, '2b': seriesex2, 'c': seriesex3, 'd': seriesex4})

print("Ex1:\n", df_ex1)

list0fdict = [
    {'a': 1, 'b': 2},
    {'b': 4, 'c': 5},
    {'a': 7, 'c': 8},
    {'c': 10, 'a': 11},
]

df_ex2 = pd.DataFrame(list0fdict, columns=["a", 'c', 'b'])
print("Ex2:\n", df_ex2)

series1 = pd.Series(np.arange(0, 1.1, 0.25), index=['a', 'aa', 'aaa', 'aaaa', 'aaaaa'])
print(series1)
dictOfSeries = series1.to_dict()

print(dictOfSeries)
df_ex3 = pd.DataFrame({"A's val": dictOfSeries})
print(df_ex3)

_2dnp_arr = np.array(np.arange(1, 10, 1.25))
_2dnp_arr = np.vstack([_2dnp_arr, _2dnp_arr[::-1]])
df_ex4 = pd.DataFrame(_2dnp_arr)
print(df_ex4)

struct_arr = np.array([(1, 2.0, 'Hello'), (2, 3.0, 'World')],
                      dtype=[('foo', 'int'), ('bar', 'float'), ('baz', 'S10')])
df_ex5 = pd.DataFrame(struct_arr)
print(df_ex5)
# %% 3. Объедините два объекта Series с неодинаковыми множествами ключей (индексов) так,
# чтобы вместо NaN было установлено значение 1

list0fdict = [
    {'a': 1, 'b': 2},
    {'b': 4, 'c': 5},
    {'a': 7, 'c': 8},
    {'c': 10, 'a': 11},
]

df_ex2 = pd.DataFrame(list0fdict, columns=["a", 'c', 'b']).fillna(1)
print("Ex2:\n", df_ex2)

# %% 4. Переписать пример с транслированием для DataFrame так, чтобы вычитание происходило по СТОЛБЦАМ
import random as rnd

a = np.array(np.arange(rnd.random(), 16), dtype=float).reshape(4, 4)
df = pd.DataFrame(a, columns=['a', 'b', 'c', 'd'])
print(df)
print(df - df["d"].values[:, np.newaxis])
# %% 5. На примере объектов DataFrame продемонстрируйте использование методов ffill() и bfill()
df_ffill_ex = pd.DataFrame([[np.nan, 2, np.nan, 0],
                            [3, 4, np.nan, 1],
                            [np.nan, np.nan, np.nan, np.nan],
                            [np.nan, 3, -1, 4]],
                           columns=list("ABCD"))

print(df_ffill_ex, "\n", "#" * 20)

df_ffill_ex.ffill(inplace=True)

print(df_ffill_ex, "\n", "#" * 20)

df_ffill_ex.bfill(inplace=True)

print(df_ffill_ex)
