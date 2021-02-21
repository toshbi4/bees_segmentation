import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

np.set_printoptions(threshold=2000000)

dir_name = os.path.dirname(__file__)
image = cv2.imread(os.path.join(dir_name, '../imgs/4bees.jpeg'))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])

plt.subplot(211), plt.imshow(gray, 'gray')
plt.subplot(212), plt.plot(histogram)
# plt.show()


# Reshape the image to a 2D array of pixels and 3 color values
Z = image.reshape((-1, 3))

# convert to np.float32
Z = np.float32(Z)

print(image.shape)
print(Z.shape)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 7
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

print(center)

# Look at determine layer
for i in range(len(center)):
    if i != 0:
        center[i] = [255, 255, 255]

# Convert back to uint8
center = np.uint8(center)
res = center[label.flatten()]
print(label.flatten().shape)
print(center.shape)
print(res.shape)
print(res)
res2 = res.reshape(image.shape)

cv2.imshow('res2', res2)
cv2.waitKey(0)
cv2.destroyAllWindows()
