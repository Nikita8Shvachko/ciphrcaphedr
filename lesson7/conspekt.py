import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# fig,ax = plt.subplots(2,3)
# fig,ax = plt.subplots(2,3,sharex = 'col',sharey='row')
# for i in range(2):
#     for j in range(3):
#         ax[i,j].text(0.5,0.5,f'({i},{j})',ha='center',va='center',fontsize=20)
#


# grid = plt.GridSpec(2,3)
#
# plt.subplot(grid[0,0])
# plt.subplot(grid[0,1:])
# plt.subplot(grid[1,:2])
# plt.subplot(grid[1,2])


# meana =[0,0]
# cova=[[1,1],[1,2]]
#
# rng = np.random.default_rng(1)
#
# x,y = rng.multivariate_normal(meana, cova, 3000).T
#
# fig = plt.figure()
# grid = plt.GridSpec(4,4,hspace=0.5,wspace=0.5)
#
# main_ax = fig.add_subplot(grid[:-1,1:])
#
# y_hist = fig.add_subplot(grid[:-1,0],xticklabels=[],sharey=main_ax)
#
# x_hist = fig.add_subplot(grid[-1,1:],yticklabels=[],sharex=main_ax)
#
# y_hist.hist(y,bins=100,orientation='horizontal',histtype='stepfilled',alpha=0.5)
#
#
# x_hist.hist(x,bins=100,histtype='step',orientation='vertical')
#
# main_ax.plot(x,y,'ok',markersize=3,alpha = 0.2)


## Поясняющие надписи

# fig,ax = plt.subplots()
# ax.text(*data,*value,*text,*style)
# ax.text(0.5,0.5,'Hello',ha='center',va='center',fontsize=20,color='red',rotation=30,transform=ax.transAxes)

# ax.set(title,xlabel,ylabel) - установка названий осей и заголовка графика

# ax.xaxis.set_major_locator(plt.NullFormatter) - убирает ось x
# ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y-%m-%d')) - устанавливает формат оси x на дату
# ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%d')) - устанавливает формат оси x на день
# ax.xaxis.set_major_locator(mpl.dates.DateFormatter('%m')) - устанавливает формат оси x на месяц
# ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(interval=10)) - устанавливает интервал оси x на 10 дней


# ax.set_xlim (0,10) - устанавливает границы оси x
# ax.text(0.5,0.5,'Hello',ha='center',va='center',fontsize=20,color='red',rotation=30,transform=ax.transAxes) - установка названий осей и заголовка графика


# fig,ax = plt.subplots()
# x = np.linspace(0,20,1000)
#
# ax.plot(x,np.cos(x))
# ax.axis('equal')
#
# ax.annotate('loc max',xy=(12.5,1),xytext=(5,2),arrowprops=dict(facecolor='red',shrink=0.01))
#
# ax.annotate('loc min',xy=(9.4,-1),xytext=(5,-5),arrowprops=dict(facecolor='black',shrink=1,width=1))


# fig,ax = plt.subplots(4,4,sharex = True,sharey=True)
# for axi in ax.flat:
#     axi.xaxis.set_major_locator(plt.MaxNLocator(4))
#     axi.yaxis.set_major_locator(plt.MaxNLocator(2))

#
# x = np.random.randn(1000)
#
# plt.hist(x)
#
# fig = plt.figure(facecolor='green')
# ax = plt.axes(facecolor='r')
#
# plt.grid(color='c', linestyle='solid')
#
# ax.xaxis.tick_bottom()
# ax.yaxis.tick_left()
# plt.style.use('default')
# plt.hist(x,bins=40,density=True,edgecolor='green')
#
# plt.show()



# .mtplotlibsrc!!!