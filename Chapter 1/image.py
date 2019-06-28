import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread("https://t1.daumcdn.net/cfile/tistory/997150355C619E031A")
# NOTE: png만 지원하나?

plt.imshow(img)
plt.show()
