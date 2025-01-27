# Pandas is an extension fo NumPy (structured arrays). Strings and rows are indexed with labels and not only number values
import numpy as np
import pandas as pd

# Series Dataframe Index

# data = pd.Series([0.25, 0.5, 0.75, 1.0])
# print(data)
# print(type(data))
#
# print(data.values)
# print(data.index)

# print(data[0])
# print(data[1:3])
#
# data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd']) # indexes can be any
# print(data)
# print(data['b':'d'])


# population_dict = {
#     'city_1': 1001,
#     'city_2': 1002,
#     'city_3': 1003,
#     'city_4': 1004,
#     'city_5': 1005,
# }
#
# population = pd.Series(population_dict)
# print(population)
#
# print(population['city_1':'city_4'])

# For Series can be used
# - lists from NumPy
# - scalar values
# - dictionaries
# todo:1
#  Привести различные способы создания объектов типа Series


## DataFrame  - 2-d array with obviously created types
#
# population_dict = {
#     'city_1': 1001,
#     'city_2': 1002,
#     'city_3': 1003,
#     'city_4': 1004,
#     'city_5': 1005,
# }
# area_dict = {
#     'city_1': 5001,
#     'city_2': 5002,
#     'city_3': 5003,
#     'city_4': 5004,
#     'city_5': 5005,
# }
#
# population = pd.Series(population_dict)
# area = pd.Series(area_dict)
#
# states = pd.DataFrame({
#     'population1': population,
#     'area1': area
# })

# print(states)
#
# print(states.values)
# print(states.index)
#
# print(states.columns)
#
# print(type(states.values))
# print(type(states.index))
#
# print(type(states.columns))


# print((states['area1']))

# DataFrame - ways to create
# - through Series
# - lists of dictionaries
# dictionaries of Series
# 2d array numpy
# structured array numpy

# todo:2
#  Привести различные способы создания объектов типа DataFrame


## Index - method to organise links to data of objects Series and Dataframee. Index - immutable and sorted,and is multiset(can be repeatable values)

# ind = pd.Index([2, 3, 5, 7, 11])
# print(ind[1])
# print(ind[::2])

# Index works on rules of set(Python)

# indA = pd.Index([1,2,3,4,5,])
# indB = pd.Index([2,3,4,5,6])
# print(indA.intersection(indB))

# Vyborka from Series

# data = pd.Series([0.25, 0.5, 0.75, 1.0],index=['a','b','c','d'])
# print('a' in data)
# print('z' in data)
#
# print(data.keys())
#
# print(list(data.items()))
#
# data['a']=100
# data['z']=1000

## as 1d array

# data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])
#
# print(data['a':'c'])
# print(data[0:2])
# print(data[(data > 0.5) & (data < 1)])

# data = pd.Series([0.25, 0.5, 0.75, 1.0], index=[1, 3, 10, 15])
# print(data[3])
#
# print(data.loc[3])
# print(data.iloc[3])


## Vyborka from DataFrame
#
# population_dict = {
#     'city_1': 1001,
#     'city_2': 1002,
#     'city_3': 1003,
#     'city_4': 1004,
#     'city_5': 1005,
# }
# area_dict = {
#     'city_1': 5001,
#     'city_2': 5002,
#     'city_3': 5003,
#     'city_4': 5004,
#     'city_5': 5005,
# }
#
# population = pd.Series(population_dict)
# area = pd.Series(area_dict)
#
# data = pd.DataFrame({'area': area, "population": population})

# print(data)
# print(data['area'])

# print(data.area)


# data['new'] = data['area']
# data['new1'] = data['area']/data['area']
# print(data)


# 2d  Numpy array

# pop1 = pd.Series({
#     'city_1': 1001,
#     'city_2': 1002,
#     'city_3': 1003,
#     'city_4': 1004,
#     'city_5': 1005,
# })
# pop2 = pd.Series({
#     'city_1': 2001,
#     'city_2': 2002,
#     'city_3': 2003,
#     'city_4': 2004,
#     'city_5': 2005,
# })
# area1 = pd.Series({
#     'city_1': 5001,
#     'city_2': 5002,
#     'city_3': 5003,
#     'city_4': 5004,
#     'city_5': 5005,
# })
#
# data = pd.DataFrame({"area": area1,'pop1': pop1, 'pop2': pop2, })
# print(data)
#
# print(data.values)
# print(data.T)
#
# print(data['area'])
#
# print(data.values[0:3])


# attributes indexers

# print(data)
#
# print(data.iloc[:3,1:2])
# print(data.loc['city_3':'city_4','pop1':'pop2'])
# print(data.loc[data['pop1']>1003,'pop1':'pop2'])
#
# data.iloc[0,2]=99999
# print(data)

#
# rng = np.random.default_rng()
#
# s = pd.Series(rng.integers(0,10,4))
#
# print(s)
# print(np.exp(s))

# #
# #
# # pop1 = pd.Series({
# #     'city_1': 1001,
# #     'city_2': 1002,
# #     'city_3': 1003,
# #     'city_4': 1004,
# #     'city_5': 1005,
# # })
# # pop2 = pd.Series({
# #     'city_1': 2001,
# #     'city_2': 2002,
# #     'city_3': 2003,
# #     'city_4': 2004,
# #     'city_5': 2005,
# # })
# # area1 = pd.Series({
# #     'city_1': 5001,
# #     'city_2': 5002,
# #     'city_3': 5003,
# #     'city_4': 5004,
# #     'city_5': 5005,
# # })
#
# data = pd.DataFrame({"area": area1,'pop1': pop1,})
#
# # todo:3


# Nan - not a number
# NA values : NaNm null, -9999

# Pandas - 2 ways of storing absent values
#Nan, None - indicators
# Null

# None - object doesnt work with sum min etc

# val1 = np.array([1,2,3])
#
# print(val1.sum())

# val1 = np.array([1,2,3,np.nan])
#
# print(val1.sum()) - # doesnt work
# print(np.nansum(val1)) - # works

# x = pd.Series(range(10),dtype=int)
# print(x)
# x[0]=None
# x[1]=np.nan
# print(x)
#
# x1 = pd.Series(['a','b','c','d'])
#
# print(x1)
# x1[0]=None
# x1[1]=np.nan
# print(x1)
# x3 = pd.Series([1,2,3,None,np.nan,pd.NA])
# print(x3)
x3 = pd.Series([11123,2,3,None,np.nan,pd.NA],dtype='Int32')
# print(x3)
#
# print(x3.isnull())
# print(x3.notna())
# print(x3[x3.notnull()])
print(x3.dropna())
x4 = pd.DataFrame([
[11123,2,3,None,np.nan,pd.NA],
    [1,2,3,4,5,6],
[11123,None,3,4,np.nan,6]
])
print(x4.dropna(axis=0))
print(x4.dropna(axis=1,thresh=3))

# how - all/any/thresh




