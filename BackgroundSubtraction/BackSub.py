from __future__ import print_function

import cv2
import cv2 as cv
import argparse
import numpy as np
from vidstab import VidStab

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
while True:
    ret, frame = capture.read()
    if frame is None:
        break

    stabilizer = VidStab()
    # Pass frame to stabilizer even if frame is None
    # frame = stabilizer.stabilize_frame(input_frame=frame, smoothing_window=30)

    # cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    # cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
    #             cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    fgMask = backSub.apply(frame)
    dst = backSub.apply(frame)
    # cv.medianBlur(dst, 5, dst)

    # apply the 3×3 mean filter on the image
    # kernel = np.ones((3, 3), np.float32) / 9
    # ImageFilter2D = cv.filter2D(dst, -1, kernel)

    """Opening"""
    # kernel = np.ones((3, 3), np.uint8)
    # opening = cv.morphologyEx(dst, cv.MORPH_OPEN, kernel, iterations=2)
    # sure_bg = cv.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    # dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    # ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # Finding unknown region
    # sure_fg = np.uint8(sure_fg)
    # unknown = cv.subtract(sure_bg, sure_fg)

    """Gaus"""
    # gauss = cv.GaussianBlur(dst, (5, 5), 0)

    """Canny"""
    # canny = cv.Canny(dst, 150, 400)

    """Connected component labeling"""
    con_comp = cv2.connectedComponentsWithStats(dst, 4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = con_comp

    ## Show Images
    # cv.imshow('Frame', frame)
    cv.imshow('BackSub', dst)
    # cv.imshow('Canny', dst)
    # cv.imshow('Gauss', gauss)
    # cv.imshow('FG Mask', fgMask)
    # cv.imshow('MedianBlur', dst)
    # cv.imshow('sure_fg', sure_fg)
    # cv.imshow('Filter2D', ImageFilter2D)

    # gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    thresh = dst
    # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # apply connected component analysis to the thresholded image
    output = cv2.connectedComponentsWithStatsWithAlgorithm(thresh, 4, cv2.CV_32S, cv2.CCL_WU)
    (numLabels, labels, stats, centroids) = output

    output = frame.copy()
    componentMask = (labels == 1).astype("uint8") * 255
    componentMask -= (labels == 1).astype("uint8") * 255
    # loop over the number of unique connected component labels
    for i in range(0, numLabels):
        # if this is the first component then we examine the
        # *background* (typically we would just ignore this
        # component in our loop)
        if i == 0:
            text = "examining component {}/{} (background)".format(i + 1, numLabels)
        # otherwise, we are examining an actual connected component
        else:
            text = "examining component {}/{}".format(i + 1, numLabels)
        # print a status message update for the current connected
        # component
        print("[INFO] {}".format(text))
        # extract the connected component statistics and centroid for
        # the current label
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]

        if (w < 500) & (h < 500) & (w > 40) & (h > 40) & ((w < 2 * h) | (h < 2 * w)):
            print(w, '  ', h)
            # clone our original image (so we can draw on it) and then draw
            # a bounding box surrounding the connected component along with
            # a circle corresponding to the centroid
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)

            # construct a mask for the current connected component by
            # finding a pixels in the labels array that have the current
            # connected component ID

            componentMask += (labels == i).astype("uint8") * 255
    # show our output image and connected component mask
    cv2.imshow("Output", output)
    if 'componentMask' in globals():
        cv2.imshow("Connected Component", componentMask)
    # cv2.waitKey(0)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
