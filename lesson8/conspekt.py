import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyparsing import alphas

# fig = plt.figure(figsize=(10, 10))
# ax = plt.axes(projection='3d')
# z = np.linspace(0, 2, 1000)
# x = np.sin(z)
# y = np.cos(z)

# ax.plot3D(x, y, z, 'gray')
#
# z2 = 10 * np.random.rand(100)
# y2 = np.cos(z2) + 0.1 * np.random.rand(100)
# x2 = np.sin(z2) + 0.1 * np.random.rand(100)


# ax.scatter3D(x2, y2, z2, c=z2, cmap='Greens')



# y = np.linspace(-6, 6, 30)
# X,Y = np.meshgrid(x,y)
#
# Z = f(X,Y)
# r = np.linspace(0, 5, 100)

# ax.contour3D(X, Y, Z, 40, cmap='binary')
#
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
#
# ax.view_init(60, 45) # - 30 градусов по вертикали, 45 градусов по горизонтали


# ax.scatter3D(X,Y,Z,c=Z,cmap='Greens')

# ax.plot_wireframe(X,Y,Z,rstride=1,cstride=1,color='green')
# ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='viridis',edgecolor='black')
# x = np.linspace(-6, 6, 30)



#
# def f(x1, y1):
#     return np.sin(np.sqrt(x1 ** 2 + y1 ** 2))
#
# theta = np.linspace(0.1* np.pi, 1.3 * np.pi ,100)
# x = r * np.cos(theta)
#
# y = r * np.sin(theta)
# z = f(x,y)
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
# ax.scatter(x,y,z, c=z, cmap='viridis')
# ax.plot(X, Y, Z, color='green')

# ax.plot_trisurf(x,y,z, cmap='viridis', edgecolor='none')

# trisurf RGB color
# def f(x, y):
#     return np.sin(np.sqrt(x ** 2 + y ** 2))
#
# x = np.linspace(-6, 6, 30)
# y = np.linspace(-6, 6, 30)
# X, Y = np.meshgrid(x, y)
# Z = f(X, Y)
# ax.plot_trisurf(X.ravel(), Y.ravel(), Z.ravel(), cmap='viridis', edgecolor='none')
# plt.show()




# Seaborn :
# - DataFrame
# - Higher level


# data = np.random.multivariate_normal([0, 0], [[5,2], [2,2]], size=2000)

# data = pd.DataFrame(data, columns=['x', 'y'])
# print(data.head())

import seaborn as sns
# sns.scatterplot(data=data, x='x', y='y', hue='x')
# plt.hist(data['x'],alpha=0.5)
# plt.hist(data['y'],alpha=0.5)
#
# fig = plt.figure(figsize=(10, 10))
#
# sns.kdeplot(data=data,fill=True)

# iris = sns.load_dataset('iris')
# print(iris.head())
#
# sns.pairplot(iris,hue = 'species')

# tips = sns.load_dataset('tips')
# print(tips.head())


## Histogram
# grid = sns.FacetGrid(tips,row='day',col='sex',hue='time')
#
# grid.map(plt.hist,'tip',bins=20)


# sns.catplot(data = tips,x = 'day',y= 'total_bill',kind = 'box')
#
# sns.jointplot(data=tips,x='total_bill',y='tip',kind='hex')


# planets = sns.load_dataset('planets')
# print(planets.head())
# sns.catplot(data=planets,x='year',kind='count',hue='method',order = [2009, 2010, 2011, 2012, 2013, 2014, 2015])
# sns.jointplot(data=planets,x='mass',y='radius',kind='hex')



tips = sns.load_dataset('tips')
print(tips.head())

# Comparison of numeric data
## Numeric pairs

# sns.pairplot(tips)



## Heat map
tips_corr = tips[['total_bill', 'tip', 'size']]

# sns.heatmap(tips_corr.corr(),cmap='RdBu_r',annot=True,vmin=-1,vmax=1)
# 0 - independent
# 1 - positive correlation
# -1 - negative correlation

# Diagram of scatter

# sns.scatterplot(data=tips,x='total_bill',y='tip',hue='sex')

# sns.regplot(data=tips,x='total_bill',y='tip',)

# sns.lmplot(data=tips,x='total_bill',y='tip',hue='sex')

# sns.relplot(data=tips,x='total_bill',y='tip',kind='line',hue='sex')

# # Linear plot

# sns.lineplot(data=tips,x='total_bill',y='tip')

# Сводная диаграмма

# sns.jointplot(data=tips,x='tip',y='total_bill')



# Comparison of numeric and categorised data

# sns.barplot(data = tips,x='day',y='total_bill',hue='sex')

# sns.pointplot(data = tips,x='day',y='total_bill',hue='sex')


# Box with whiskers

# sns.boxplot(data=tips,x='day',y='total_bill',hue='sex')


# # Violin plot

# sns.violinplot(data=tips,x='day',y='total_bill',hue='sex')


# # 1d strip diagram

sns.stripplot(data=tips,x='day',y='total_bill',hue='sex')



plt.show()










