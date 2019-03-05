import checkersAnalyzer as ca
import cv2
import numpy as np

if __name__ == '__main__':
    cap = cv2.VideoCapture('Picture\movie.avi')
    analyzer = ca.checkersAnalyzer(False)
    ret, frame = cap.read()
    analyzer.readVideo(frame)
    analyzer.detectAreaBoardDistribution()
    analyzer.intoDictionary()
    while (cap.isOpened()):
        analyzer.detectCircle()
        b, c, d = analyzer.drawBoard()
        cv2.imshow('Zaznaczone pionki', b)
        cv2.imshow('Plansza', c)
        cv2.imshow('Wizualizacja', d)
        ret, frame = cap.read()
        analyzer.readVideo(frame)
        cv2.waitKey(20)
cap.release()
cv2.destroyAllWindows()