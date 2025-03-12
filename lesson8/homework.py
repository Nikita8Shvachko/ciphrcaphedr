import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import math as m

#%% 1
x1 = np.linspace(1, 20, 5)
x2 = np.linspace(1, 20, 5)

y1 = np.random.randint(1,13,5)
y2 = np.random.randint(1,13,5)
fig1 = plt.plot(figsize=(10, 10))
print(x1,x2,y1,y2)

plt.plot(x1,y1, marker='o', linestyle='--', color='g')
plt.plot(x2,y2, marker='o', linestyle='-', color='r')
plt.xlim(0,20)
plt.ylim(0,12)

plt.legend(['line1', 'line2'], loc='upper left', fancybox=True, shadow=True)
plt.show()


# %% 2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import math as m
x = np.linspace(1,5,5)

def f (x):
    return 1.2*(x-3)**2+3
y1 = np.random.randint(2,6,5)
y2 = f(x)
y3_1 = np.linspace(-7,2,3)
y3_2 =y3_1[0:2]
y3 = np.append(y3_1,y3_2[::-1])

fig = plt.figure(figsize=(20, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 2])

ax1 = fig.add_subplot(gs[0, :])
ax1.plot(x,y1)
plt.xlim(1,5)
plt.tick_params(axis='both', which='major', labelsize=15)

ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(x, y2,)
plt.xlim(1,5)
plt.ylim(2,8)
plt.tick_params(axis='both', which='major', labelsize=15)

ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(x, y3)
plt.xlim(1,5)
plt.ylim(-6,2)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.show()


# %% 3***
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 11)
y = x*x

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.xlim(-5.5, 5.5)
ymin = np.min(y)
imin = np.argmin(y)
plt.annotate("",(x[imin],ymin),( 0, 6),arrowprops=dict( facecolor='g', edgecolor='b') )
plt.text(x[imin], ymin+6, 'min', fontsize=15)
plt.show()


# %% 3
import matplotlib.pyplot as plt
import numpy as np


xy = np.random.randint(0, 11, (7, 7))
print(xy, '\n')
fig = plt.figure(figsize=(10, 10))
ax = plt.axes()
ax.imshow(xy, cmap='viridis')
cax = fig.colorbar(ax.imshow(xy, cmap='viridis'), ax=ax, location='right', aspect=3.5, shrink=0.2,
                   ticks=[0, 2, 4, 6, 8, 10])
cax.mappable.set_clim(0, 10)

plt.show()


# %% 4
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return np.cos(x * np.pi)


x = np.linspace(0, 5, 1000)
y = f(x)
fig = plt.figure(figsize=(10, 4))
ax = plt.axes()
ax.plot(x, y, color='r', linewidth=3)
ax.fill_between(x, y)

plt.show()


# %% 5

import matplotlib.pyplot as plt
import numpy as np


def f(x):
    res = np.cos(x * np.pi)
    return res


x = np.linspace(0, 5, 1000)
y = f(x)
y[y < -0.5] = np.nan
fig = plt.figure(figsize=(10, 4))
ax = plt.axes()
ax.plot(x, y)
plt.ylim(-1,1)

plt.show()


# %% 6

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 7)
y = np.arange(0, 7)

fig, ax = plt.subplots(1, 3, figsize=(10, 3))

wheres = ["pre", "post", "mid"]
for i, wheres_i in enumerate(wheres):
    plt.subplot(1, 3, i + 1)
    plt.title(wheres_i)
    plt.step(x, y, where=wheres_i, color='g', marker='o', )
    plt.grid(True)
    plt.xlim(-0.5, 6.5)
    plt.ylim(-0.5, 6.5)
    plt.xticks(np.arange(0, 7))
    plt.yticks(np.arange(0, 7))


plt.show()
print(x)


# %% 7

import matplotlib.pyplot as plt
import numpy as np


x = np.arange(0, 10, 0.1)
y1 = -0.25 * x*(x-10)
y2 = -0.5 * x*(x-10)
y3 = -0.7 * x*(x-12)

plt.plot(figsize=(10, 10))
plt.plot(x, y1, label='y1')
plt.plot(x, y2, label='y2')
plt.plot(x, y3, label='y3')
plt.legend(loc='upper left')

plt.fill_between(x, 0, y1)
plt.fill_between(x, y1, y2)
plt.fill_between(x, y2, y3)
plt.ylim(0)


plt.show()

# %% 8
import matplotlib.pyplot as plt
import numpy as np

bmw = 65
audi = 30
toyota = 20
jaguar = 40
ford = 30

labels = ['bmw', 'audi', 'jaguar', 'ford', 'toyota']
values = [bmw, audi, jaguar, ford, toyota]
explode = [0.2, 0, 0, 0, 0]
colors = ["tab:green", "tab:red", "tab:purple", "tab:blue", "tab:orange"]

fig, ax = plt.subplots(figsize=(10, 10))
ax.pie(values, explode=explode, labels=labels, colors=colors, startangle=90)
ax.axis('equal')

plt.show()

# %% 9
import matplotlib.pyplot as plt
import numpy as np

bmw = 65
audi = 30
toyota = 20
jaguar = 40
ford = 30

labels = ['bmw', 'audi', 'jaguar', 'ford', 'toyota']
values = [bmw, audi, jaguar, ford, toyota]
colors = ["tab:green", "tab:red", "tab:purple", "tab:blue", "tab:orange"]
plt.subplots(figsize=(10, 10))
plt.pie(values, labels=labels, colors=colors, startangle=90,wedgeprops=dict(width=0.5, edgecolor='w'))
plt.axis('equal')

plt.show()

