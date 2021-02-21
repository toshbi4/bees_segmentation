#!/usr/bin/env python3

"""

    Potapov Anton
    22.11.2020
    OpenCv Bees segmentation with Water Shed algorithm

"""

import os
import numpy as np
import cv2

# read in images
dir_name = os.path.dirname(__file__)
image = cv2.imread(os.path.join(dir_name, '../imgs/multiple_bees.jpg'))

# convert to grayscale
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown == 255] = 0
markers_labels = cv2.watershed(image, markers)
print(markers_labels)
image[markers == -1] = [255, 0, 0]

cv2.imshow("Water Shed", image)

while True:
    if cv2.waitKey(10) == 27:  # Esc key
        break

cv2.destroyAllWindows()
