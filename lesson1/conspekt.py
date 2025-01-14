# import sys
# Data types
#
# x = 1
# print(sys.getsizeof(x))
#
# x = 'Hello'
# print(type(x))
#


## Todo: 1
## Todo: 2


# a = np.array([1.22, 2, 3], dtype=np.int16)
# print(type(a),a)

# a = np.array([range(i, i + 3) for i in [2, 4, 6]], dtype=np.int16)
# print(type(a),a)

# a = np.zeros((3, 4), dtype=np.int16)
# print(type(a),a)

# b = np.ones((3, 4), dtype=np.int16)
# print(type(b),b)

# c = np.full((3, 4), 3.14, dtype=np.float16)
# print(type(c),c)

# d = np.eye(4, dtype=np.int16)
# print(type(d),d)

# print(np.arange(10))

## Todo: 3
## Todo: 4
## Todo: 5
## Todo: 6


## Arrays

# np.random.seed(1)
# x1 = np.random.randint(10, size=3)
# x2 = np.random.randint(10, size=(3,2))
# print(x2.ndim, x2.shape, x2.size)


# Index from (0)
# a = np.array([1, 2, 3, 4, 5])
# print(a[0])
# print(a[-3])

# a = np.array([[1,2], [3,4]])
# print(a)
#
# a[0, 1] = 10
# print(a)

## Slicing: [start:end:step] from (0)

# a = np.array([1, 2, 3, 4, 5,6])
# print(a[0:3:1])
# print(a[:3])
# print(a[3:])
# print(a[1:-1:])
# print(a[::-3])

## Todo: 7
## Todo: 8
# a = np.arange(1,13)
# print(a)
# print(a.reshape(2,6))
# print(a.reshape(3,4))
## Todo: 9


# x = np.array([1, 2, 3])
# y = np.array([4, 5,6])
# z = np.array([6])
# print(np.concatenate((x, y, z)))

# r1 = np.vstack([x, y])
# r2 = np.hstack([x, y])
# print(r1)
# print(r2)
## Todo: 10
## Todo: 11


### Calculations using arrays
# Vectorizated operations - independent for each element

# x = np.arange(10)
# print(x)
# print(x*2+1)

# Universal functions

# print(np.add(np.multiply(x, 2),11))
# - - / // % **
# np.abs np.ceil np.floor np.round np.sqrt  np.log  np.exp  np.sin  np.cos  np.tan
## Todo: 12


# x = np.arange(5)
# y = np.zeros(10)
# print(np.multiply(x, 10,out = y[::2]))
# print(y)

# x = np.arange(1,5)
# print(x)
# print(np.add.reduce(x))
# print(np.add.accumulate(x))

# x = np.arange(1,10)
# print(np.add.outer(x, x))
# print(np.multiply.outer(x, x))
