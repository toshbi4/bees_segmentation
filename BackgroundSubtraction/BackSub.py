#!/usr/bin/env python3
from __future__ import print_function

import cv2
import cv2 as cv
import argparse
import math
import time

"""Constants"""
# the more splitting parts the more accuracy, but the longer calculations
splitting_parts = 10
# FrameMask with trajectories
trajectories = None
# FrameMask with Text
textMask = None
# First square
refSquare = 1000
"""Constants"""

db = open('databases.txt', 'w')
parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()

if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG(1, 16, True)
else:
    backSub = cv.createBackgroundSubtractorKNN()

capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)

velFrameCount = 0
beesLabels = [[], []]
prevCentroids = []
frameCount = 0

while True:
    ret, frame = capture.read()
    frameCount += 1
    if frame is None:
        break

    if trajectories is None:
        trajectories = frame.copy()
        trajectories[:] = (0, 0, 0)
    if textMask is None:
        textMask = frame.copy()
        textMask[:] = (0, 0, 0)

    fgMask = backSub.apply(frame)
    dst = backSub.apply(frame)

    """Connected component labeling"""
    con_comp = cv2.connectedComponentsWithStats(dst, 4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = con_comp

    # Show Images

    # cv.imshow('BackSub', dst)

    thresh = dst

    # apply connected component analysis to the thresholded image
    output = cv2.connectedComponentsWithStatsWithAlgorithm(thresh, 4, cv2.CV_32S, cv2.CCL_WU)
    (numLabels, labels, stats, centroids) = output

    output = frame.copy()
    componentMask = (labels == 1).astype("uint8") * 255
    componentMask -= (labels == 1).astype("uint8") * 255

    # loop over the number of unique connected component labels
    for i in range(0, numLabels):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]

        if (w < 500) & (h < 500) & (w > 40) & (h > 40) & ((w < 2 * h) | (h < 2 * w)):
            # clone our original image (so we can draw on it) and then draw
            # a bounding box surrounding the connected component along with
            # a circle corresponding to the centroid
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)

            # construct a mask for the current connected component by
            # finding a pixels in the labels array that have the current
            # connected component ID

            componentMask += (labels == i).astype("uint8") * 255
            beesLabels[velFrameCount].append(i)

    if beesLabels[velFrameCount]:
        if velFrameCount < 1:
            prevCentroids = centroids
            prevStats = stats
            prevTime = time.time()
            velFrameCount += 1
        else:

            for k1 in beesLabels[1]:
                (cX1, cY1) = centroids[k1]
                minTrace = 1000000
                nearestLabel = None
                for k0 in beesLabels[0]:
                    (cX0, cY0) = prevCentroids[k0]
                    trace = math.sqrt((cX1 - cX0)**2 + (cY1 - cY0)**2)
                    if minTrace > trace:
                        minTrace = trace
                        nearestLabel = k0
                if (prevCentroids.any()) & (nearestLabel is not None):
                    (cX0, cY0) = prevCentroids[nearestLabel]
                    if minTrace > 100:
                        break

                    deltaTrace = minTrace / splitting_parts
                    square0 = prevStats[k0, cv2.CC_STAT_WIDTH] * prevStats[k0, cv2.CC_STAT_HEIGHT]
                    squareN = stats[k1, cv2.CC_STAT_WIDTH] * stats[k1, cv2.CC_STAT_HEIGHT]

                    size_0 = square0 / refSquare
                    size_N = squareN / refSquare

                    deltaSize = (size_N - size_0) / splitting_parts

                    currentTime = time.time()

                    length = 0 # length of route between two frames
                    for j in range(splitting_parts):
                        length = length + deltaTrace / (deltaSize * j + size_0)

                    if currentTime > prevTime:
                        vel = length / (currentTime - prevTime)

                    length = length + abs(1.0 / size_N - 1.0 / size_0)

                    x_text = stats[k1, cv2.CC_STAT_LEFT]
                    y_text = stats[k1, cv2.CC_STAT_TOP]

                    db.write(str(frameCount) + ": {:.2f}".format(vel) + '\n')

                    cv2.putText(textMask, "{:.2f}".format(vel), (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                    cv2.line(trajectories, (int(cX1), int(cY1)), (int(cX0), int(cY0)), (255, 255, 255), 5)
                    beesLabels[0].remove(nearestLabel)

            # Shift the list by one frame
            del beesLabels[0]
            beesLabels.append([])
            prevCentroids = centroids
            prevStats = stats
            prevTime = time.time()

    # show our output image and connected component mask

    cv2.imshow("OutputALL", cv2.add(trajectories, cv2.add(textMask, output)))

    textMask[:] = (0, 0, 0)
    # if 'componentMask' in globals():
        # cv2.imshow("Connected Component", componentMask)
    # cv2.waitKey(0)

    keyboard = cv.waitKey(30)

    # Pause
    print(keyboard)
    if keyboard == 112:
        print(keyboard)
        while 1:
            keyboard = cv.waitKey(30)
            print(keyboard)
            if keyboard == 115:
                break

    if keyboard == 'q' or keyboard == 27:
        break
