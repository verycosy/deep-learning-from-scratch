import numpy as np

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
print(x)
print(type(x))
print(x.dtype)
print(x+y)

A = np.array([[1,2],[3,4]])
B = np.array([10,20])

print(A.shape)
print(A*B)

for row in A:
    print(row)

A = A.flatten()
print(A)

print(A[np.array([0,3])])

print(A > 2)
print(A[A>2])