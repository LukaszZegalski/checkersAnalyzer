import cv2 as cv
import numpy as np


class checkersAnalyzer(object):
    def __init__(self, debug):
        self.image = None
        self.debug = debug
        self.points = []

    def read(self, path):
        self.image = cv.imread(path, cv.IMREAD_GRAYSCALE)
        self.image = cv.resize(self.image, (300,300))

        if self.debug:
            cv.imshow('Orginal', self.image)

    def threshlod(self):
        def onMouse(event, x, y, flags, param):
            if event == cv.EVENT_LBUTTONDOWN:
                print('x = %d, y = %d' % (x, y))
                self.points.append([x,y])

        threshBinary = self.image
        threshBinary = cv.blur(threshBinary, (5, 5))
        _, threshBinary = cv.threshold(threshBinary, 110, 255, cv.THRESH_BINARY)
        cv.imshow('Binary threshold', threshBinary)
        cv.setMouseCallback('Binary threshold', onMouse)

        cv.waitKey(0)
        p = self.points
        pts1 = np.float32([[p[0][0], p[0][1]], [p[1][0], p[1][1]], [p[2][0], p[2][1]], [p[3][0], p[3][1]]])
        pts2 = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])

        M = cv.getPerspectiveTransform(pts1, pts2)

        dst = cv.warpPerspective(threshBinary, M, (300, 300))

        cv.imshow('Warp', dst)
        cv.waitKey(0)



