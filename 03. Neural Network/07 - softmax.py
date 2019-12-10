import numpy as np

a = np.array([0.3, 2.9, 4.0])

def softmax(a):
    c = np.max(a)

    exp_a = np.exp(a-c) # Overflow 대책
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

print(softmax(a))