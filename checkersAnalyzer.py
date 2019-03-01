import cv2 as cv

class checkersAnalyzer(object):

    def __init__(self, debug):
        self.image = None
        self.debug = debug

    def read(self, path):
        self.image = cv.imread(path, 0)

        if self.debug:
            cv.imshow('Orginal', self.image)

    def threshold(self):
        if self.debug:
            thresType = ['Binary', 'Binary inv', 'Trunc', 'Tozero', 'Tozero inv']
            result = []
            ret, thresh1 = cv.threshold(self.image, 127, 255, cv.THRESH_BINARY)
            result.append(thresh1)
            ret, thresh1 = cv.threshold(self.image, 127, 255, cv.THRESH_BINARY_INV)
            result.append(thresh1)
            ret, thresh1 = cv.threshold(self.image, 127, 255, cv.THRESH_TRUNC)
            result.append(thresh1)
            ret, thresh1 = cv.threshold(self.image, 127, 255, cv.THRESH_TOZERO)
            result.append(thresh1)
            ret, thresh1 = cv.threshold(self.image, 127, 255, cv.THRESH_TOZERO_INV)
            result.append(thresh1)

            for i in range(0, len(result)):
                cv.imshow(thresType[i], result[i])

            cv.waitKey(0)

    def shapeDetection(self):
        shape = "unidentified"
        peri = cv.arcLength(self.image, True)
        approx = cv.approxPolyDP(self.image, 0.04 * peri, True)

