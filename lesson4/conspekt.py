# MultIndex

import numpy as np
import pandas as pd

print("\n" * 3)

# %% Pseudo MultiIndex
# print("\n" * 3)
# population = [
#     1001,
#     1002,
#     1003,
#     1004,
#     1005,
#     1006,
# ]
# index = [
#     ('city_1', 2010),
#     ('city_1', 2011),
#     ('city_2', 2012),
#     ('city_2', 2013),
#     ('city_3', 2014),
#     ('city_3', 2015),
# ]
# pop = pd.Series(population, index=index)
# print(pop)
# print(pop[[i for i in pop.index if i[1] == 2020]])
#
# # %% MultiIndex
# print("\n" * 3)
# index = pd.MultiIndex.from_tuples(index)
# pop = pop.reindex(index)
# print(pop)
#
# print(pop[:, 2010])
#
# pop_df = pop.unstack()
# print(pop_df.stack())

# %%
# print("\n" * 3)
# population = [
#     1001,
#     10012,
#
#     1003,
#     10014,
#
#     1005,
#     10016,
#
#     1001,
#     10012,
#
#     1003,
#     10014,
#
#     1005,
#     10016,
# ]
#
# index1 = [
#     ('city_1', 2010, 1),
#     ('city_1', 2010, 2),
#
#     ('city_1', 2011, 1),
#     ('city_1', 2011, 2),
#
#     ('city_2', 2012, 1),
#     ('city_2', 2012, 2),
#
#     ('city_3', 2013, 1),
#     ('city_3', 2013, 2),
#
#     ('city_2', 2014, 1),
#     ('city_2', 2014, 2),
#
#     ('city_3', 2015, 1),
#     ('city_3', 2015, 2),
# ]
# pop = pd.Series(population, index=index1)
# print(pop)
#
# index1 = pd.MultiIndex.from_tuples(index1)
# pop = pop.reindex(index1)
# print(pop)
# print(pop[:, :, 1])
# print(pop[:, :, 2])
#
# pop_df = pop.unstack()
# print(pop_df)
# print(pop_df.stack())

# %%
# population = [
#     1001,
#     10012,
#
#     1003,
#     10014,
#
#     1005,
#     10016,
#
#     1001,
#     10012,
#
#     1003,
#     10014,
#
#     1005,
#     10016,
# ]
# index = [
#     ('city_1', 2010),
#     ('city_1', 2011),
#
#     ('city_2', 2012),
#     ('city_2', 2013),
#
#     ('city_3', 2014),
#     ('city_3', 2015),
#
#     ('city_1', 2010),
#     ('city_1', 2011),
#
#     ('city_2', 2012),
#     ('city_2', 2013),
#
#     ('city_3', 2014),
#     ('city_3', 2015),
# ]
# pop = pd.Series(population, index=index)
# pop_df = pd.DataFrame({
#     'total': pop,
#     'something': [
#         10,
#         11,
#         12,
#         13,
#         14,
#         15,
#         16,
#         17,
#         18,
#         19,
#         20,
#         21
#     ]
# })
# print(pop_df)
#
#
#
# #  - список кортежей задающий индекс в каждом точке
# i2 = pd.MultiIndex.from_tuples(
#     [
#         ('city_1', 2010),
#         ('city_1', 2011),
#
#         ('city_2', 2012),
#         ('city_2', 2013),
#
#     ]
# )
# print(i2)

#  Декартово произведение индексов

# i3 = pd.MultiIndex.from_product(
#     [
#         ['city_1', 'city_2'],
#         [2010, 2011]
#     ]
# )
# print(i3)

# Описание внутреннего представления: levels, codes

# i4 = pd.MultiIndex(
#     levels=[['city_1', 'city_2', 'city_3'], [2010, 2011, 2012, ]],
#     codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]], )
# print(i4)
#
#
# # Уровням можно присвоить имена
# data = {
#     ('city_1', 2010): 1001,
#     ('city_1', 2011): 1002,
#     ('city_2', 2010): 1003,
#     ('city_2', 2011): 1004
# }
# s = pd.Series(data)
# print(s)
# s.index.names = ['city', 'year']
# print(s)

# index = pd.MultiIndex.from_product(
#     [['city_1', 'city_2',],
#      [2010, 2020]],
#     names=['city', 'year']
# )
# print(index)
#
# columns = pd.MultiIndex.from_product(
#     [
#         ['person_1', 'person_2', 'person_3'],
#         ['job_1', 'job_2']
#     ],
#     names=['worker', 'job'])
#
# rng = np.random.default_rng(1)
# data = rng.random((4, 6))
# df = pd.DataFrame(data, index=index, columns=columns)
# print(df)


# Индексация и срезы по мультииндексу

# data = {
#     ('city_1', 2010): 1001,
#     ('city_1', 2011): 1002,
#     ('city_2', 2010): 1003,
#     ('city_2', 2011): 1004
# }
#
# s = pd.Series(data)
# s.index.names = ['city', 'year']
# print(s['city_1', 2010])
# print(s[s>=1002])


# Перегрупировка мультииндексов
rng = np.random.default_rng(1)
# index = pd.MultiIndex.from_product(
#     [
#         ['a', 'c', 'b'],
#         [1, 2]
#     ]
# )
# data = pd.Series(rng.random(6), index=index)
# data.index.names = ['char', 'int']
#
# print(data)
# # print(data['a':'b'])
# data = data.sort_index()
# print(data)


# index = [
#     ('city_1', 2010, 1),
#     ('city_1', 2010, 2),
#
#     ('city_3', 2011, 1),
#     ('city_3', 2011, 2),
#
#     ('city_2', 2012, 1),
#     ('city_2', 2012, 2),
#
# ]
# population = [
#     1001,
#     10012,
#
#     1003,
#     10014,
#
#     1005,
#     10016
# ]
#
# pop = pd.Series(population, index=index)
# print(pop)
#
# i = pd.MultiIndex.from_tuples(index)
# pop = pop.reindex(i)
# print(pop)
# print(pop.unstack())
# print(pop.unstack(level=0))
# print(pop.unstack(level=1))
# print(pop.unstack(level=2))

# NumPy конкатенация
# a = [[1, 2, 3]]
# b = [[4, 5, 6]]
# print(np.concatenate([a, b], axis=0))

# Pandas конкатенация
a = pd.DataFrame([[1, 2, 3]])
b = pd.DataFrame([[4, 5, 6]])
print(pd.concat([a, b], axis=0))
print(pd.concat([a, b], axis=1))

ser1 = pd.Series(['a', 'b', 'c'], index=[1, 2, 3])
ser2 = pd.Series(['c', 'c', 'f'], index=[4, 3, 6])
ser3 = pd.Series(['c', 'c', 'f'], index=[4, 3, 6])

# print(pd.concat([ser1, ser2], verify_integrity=False))
# print(pd.concat([ser1, ser2], ignore_index=True))
# print(pd.concat([ser1, ser2], keys=['ser1', 'ser2']))

print(pd.concat([ser1, ser2, ser3], join='inner'))
print(pd.concat([ser1, ser2, ser3], join='outer'))
