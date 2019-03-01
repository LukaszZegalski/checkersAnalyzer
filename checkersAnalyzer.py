import cv2 as cv
import numpy as np

class checkersAnalyzer(object):

    def __init__(self, debug):
        self.image = None
        self.debug = debug

    def read(self, path):
        self.image = cv.imread(path)
        self.image = cv.resize(self.image, (300,300))

        if self.debug:
            cv.imshow('Orginal', self.image)

    def detectBoard(self):
        edges = cv.Canny(self.image, 50, 150, apertureSize=3)
        lines = cv.HoughLinesP(image=edges, rho=1, theta=np.pi / 180, threshold=100, lines=np.array([]),
                                minLineLength=100, maxLineGap=80)
        i = 0
        cv.line(self.image, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv.LINE_AA)
        i = len(lines) - 1
        cv.line(self.image, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv.LINE_AA)


        # Todo: Need calculate coordinate of right upper point.
        if self.debug:
            cv.imshow('Lines', self.image)
            cv.waitKey(0)

    def shapeDetection(self):
        shape = "unidentified"
        peri = cv.arcLength(self.image, True)
        approx = cv.approxPolyDP(self.image, 0.04 * peri, True)

