import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
rng = np.random.default_rng(1)
data = rng.normal(size=1000)

# plt.hist(data,
#          bins = 40,
#          density=True,
#          alpha=0.5,
#          histtype='stepfilled',
#          edgecolor='red',
#          )


x1 = rng.normal(0, 0.8, size=1000)
x2 = rng.normal(-2, 1, size=1000)
x3 = rng.normal(3, 2, size=1000)

# args = dict(
#     alpha = 0.3,
#     bins = 40,
# )
#
# plt.hist(x1, **args)
# plt.hist(x2, **args)
# plt.hist(x3, **args)

# plt.show()


# print(np.histogram(x1,bins=1))
# print(np.histogram(x1,bins=2))
# print(np.histogram(x1,bins=40))


# 2D Histogram
# mean = [0,0]
# cov = [[1,1],[1,2]]
# x,y = np.random.multivariate_normal(mean,cov,10000).T
#
# plt.hexbin(x,y,bins=20)

# plt.hist2d(x,y,bins=40, density=True)
# plt.colorbar()
# plt.show()
#
# print(np.histogram2d(x,y,bins=40))


# Legend

# x = np.linspace(0, 10, 1000)
# fig, ax = plt.subplots()
#
# ax.plot(x, np.sin(x), label='sin')
# ax.plot(x, np.cos(x), label='cos')
# ax.plot(x, np.cos(x) + 2)
# ax.axis('equal')
#
# ax.legend(
#     frameon=True,
#     shadow=True,
#     fancybox=True,
#     loc='best',
# )
# plt.legend(['1q', 'second', 'third'])
# plt.show()
# seaborn - костыль для графиков в матплотлиб через pandas




# fig,ax = plt.subplots()
# lines = []
# styles = ['-', '--', '-.', ':']
# x = np.linspace(0, 10, 1000)
# for i in range(4):
#     lines += ax.plot(
#         x,
#         np.sin(x - i+np.pi/2),
#         styles[i]
#     )
#
#
# ax.legend(lines, ['sin', 'sin+1', 'sin+2', 'sin+3'])
#
# ax.axis ('equal')
# ax.legend(lines[:2],['sin', 'sin+1'], loc='upper left')
#
# leg = mpl.legend.Legend(ax, lines[1:],['sin+2', 'sin+3'], loc='upper right')
# ax.add_artist(leg)
#
# plt.show()


# Scales

x = np.linspace(0, 10, 1000)
y = np.sin(x)*np.cos(x[:,np.newaxis])

# plt.imshow(y,cmap='hot')
# plt.colorbar()

# Color maps:
# 1 последовательные
# 2 дивергентные
# 3 - качественные

#1
# plt.imshow(y,cmap='viridis')
# plt.imshow(y,cmap='binary')

#2
# plt.imshow(y,cmap='RdBu')
# plt.imshow(y,cmap='PuOr')

# 3
# plt.imshow(y,cmap='rainbow')
# plt.imshow(y,cmap='jet')
#

# Также можно сделать color map дискретным

plt.axes([0.1, 0.1, 0.8, 0.8]) # отступы от краев графика [left, bottom, width, height]






plt.show()
