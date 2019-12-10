import numpy as np

arr = np.array([1,2,3,4])
print(np.ndim(arr)) # 1
print(arr.shape) # (4,)
# 2차원, 3차원 배열일 때와 통일된 형태로 결과를 반환하기 위해 1차원 배열이라도 튜플을 반환한다.

print(arr.shape[0]) # 4

A = np.array([[1,2,3],[4,5,6]]) # 2행 3열
B = np.array([[1,2],[3,4], [5,6]]) # 3행 2열

print(np.dot(A,B)) # 2X3 * 3X2 = 2*2 (내적, 행렬곱)

C = np.array([7,8])
print(C.shape) # (2,)

print(np.dot(B,C)) # 3X2 * 2 = 3   [23, 53, 83]

# 어떤 경우든 다차원 배열을 곱하려면 두 행렬의 대응하는 차원의 원소 수를 일치시켜야 한다.

