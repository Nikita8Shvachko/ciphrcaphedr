import numpy as np
import pandas as pd

# %% # 1. Разобраться как использовать мультииндексные ключи в данном примере
index = pd.MultiIndex.from_tuples([
    ('city_1', 2010),
    ('city_1', 2020),
    ('city_2', 2010),
    ('city_2', 2020),
    ('city_3', 2010),
    ('city_3', 2020),
])

population = [
    101,
    201,
    102,
    202,
    103,
    203,
]
pop = pd.Series(population, index=index)
pop_df = pd.DataFrame(
    {
        'total': pop,
        'something': [
            10,
            11,
            12,
            13,
            14,
            15,
        ]
    }
)
print(pop_df)
# ???? ## pop_df_1 = pop_df.loc???['city_1', 'something']
pop_df_1 = pop_df.loc[('city_1', 2010), 'something']
print(pop_df_1)
# ???? ## pop_df_1 = pop_df.loc???[['city_1', 'city_3'], ['total', 'something']]
pop_df_2 = pop_df.loc[(['city_1', 'city_3'], slice(None)), ['total', 'something']]
print(pop_df_2)
# ???? ## pop_df_1 = pop_df.loc???[['city_1', 'city_3'], 'something']
pop_df_3 = pop_df.loc[(['city_1', 'city_3'], slice(None)), 'something']
print(pop_df_3)

# %% 2. Из получившихся данных выбрать данные по
data = {
    ('city_1', 2010): 1001,
    ('city_1', 2011): 1002,
    ('city_2', 2010): 1003,
    ('city_2', 2011): 1004
}
s = pd.Series(data)
s.index.names = ['city', 'year']

index = pd.MultiIndex.from_product(
    [['city_1', 'city_2', ],
     [2010, 2020]],
    names=['city', 'year']
)

columns = pd.MultiIndex.from_product(
    [
        ['person_1', 'person_2', 'person_3'],
        ['job_1', 'job_2']
    ],
    names=['worker', 'job'])
rng = np.random.default_rng(12)
data = (rng.random((4, 6)))
data = (data*100)//10

df = pd.DataFrame(data, index=index, columns=columns)
# %%
print(df)
# %% - 2020 году (для всех столбцов)
print(df.loc[:, 2020, :])
# %% - job_1 (для всех строк)
print(df.xs('job_1', axis=1, level='job'))
# %% - для city_1 и job_2
print(df.loc['city_1', (slice(None), 'job_2')])

# %% 3. Взять за основу DataFrame со следующей структурой
index = pd.MultiIndex.from_product(
    [
        ['city_1', 'city_2'],
        [2010, 2020]
    ],
    names=['city', 'year']
)
columns = pd.MultiIndex.from_product(
    [
        ['person_1', 'person_2', 'person_3'],
        ['job_1', 'job_2']
    ],
    names=['worker', 'job']
)
rng = np.random.default_rng(1)
data = rng.random((4, 6))

# Создание DataFrame
df = pd.DataFrame(data, index=index, columns=columns)
print('#'*90)

print(df)
print('#'*90)
#
# Выполнить запрос на получение следующих данных
# - все данные по person_1 и person_3
df_person_1_3 = df.loc[:, pd.IndexSlice[['person_1', 'person_3'], :]]
print(df_person_1_3)
print('-#'*90)

# - все данные по первому городу и первым двум person-ам (с использование срезов)
df_city_1_first_2 = df.loc['city_1', pd.IndexSlice[:'person_2', :]]
print(df_city_1_first_2)
print('#-'*90)

# Приведите пример (самостоятельно) с использованием pd.IndexSlice


# %% 4. Привести пример использования inner и outer джойнов для Series (данные примера скорее всего нужно изменить)
import pandas as pd
import numpy as np

print('#'*90)
ser1 = pd.Series(['a', 'b', 'c'], index=[1,2,3])
ser2 = pd.Series(['x', 'y', 'z'], index=[3,4,5])
ser3 = pd.Series(['pa', 'qd', 're'], index=[6,7,8])
print("axis=0, \n1) inner")
print (pd.concat([ser1, ser2, ser3], axis=0,join='inner'))
print("2) outer")
print (pd.concat([ser1, ser2, ser3], axis=0,join='outer'))
print('|'*90)

print("axis=1, \n1) inner")
print (pd.concat([ser1, ser2, ser3], axis=1,join='inner'))
print("2) outer")
print (pd.concat([ser1, ser2, ser3], axis=1,join='outer'))

