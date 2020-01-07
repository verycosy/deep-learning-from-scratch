import sys, os

sys.path.append(os.pardir)
import pickle
import numpy as np
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(
        flatten=True, normalize=True, one_hot_label=False
    )
    # 훈련 이미지/레이블, 시험 이미지/레이블
    # normalize : True - 입력 이미지 픽셀값 0.0~1.0 사이로 정규화, False - 원래 값 그대로 0~255
    # flatten : True - 1차원 배열, Flase - 1x28x28
    # one_hot_label : 원-핫 인코딩 형태로 저장할지. True - 한 원소만 1이고 나머지는 0, False - 숫자 라벨 그대로.

    print(x_train.shape)
    print(t_train.shape)
    print(x_test.shape)
    print(t_test.shape)

    img = x_train[0]
    label = t_train[0]
    print(label)

    print(img.shape)
    img = img.reshape(28, 28)
    print(img.shape)

    img_show(img)  # normalize=False 일 때 확인 가능

    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    # W1.shape = (784,50)
    # W2.shape = (50,100)
    # W3.shape = (100,10)

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()

accuracy_cnt = 0

"""
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)

    if p == t[i]:
        accuracy_cnt += 1
"""

batch_size = 100

for i in range(0, len(x), batch_size):
    x_batch = x[i : i + batch_size]  # 100 x 784
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)

    accuracy_cnt += np.sum(p == t[i : i + batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
