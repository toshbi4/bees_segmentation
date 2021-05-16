from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
args = parser.parse_args()

capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)

history = []
while True:
    ret, frame = capture.read()
    if frame is None:
        break
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    history.append(frame)

    img = []
    if len(history) > 5:
        a1 = np.array(history[len(history) - 1])
        a2 = np.array(history[len(history) - 2])
        a3 = np.array(history[len(history) - 3])
        a4 = np.array(history[len(history) - 4])
        a5 = np.array(history[len(history) - 5])
        img = a1-a2-a3-a4-a5
        print(img)

        ## Show Images
        cv.imshow('Frame', img)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
