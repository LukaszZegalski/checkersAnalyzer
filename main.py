import checkersAnalyzer as ca
import cv2

if __name__ == '__main__':
    a = ord('a')
    analyzer = ca.checkersAnalyzer(False)
    cap = cv2.VideoCapture('Picture/movie2.mp4')
    ret, frame = cap.read()
    analyzer.readVideo(frame)
    analyzer.detectAreaBoardDistribution()
    analyzer.intoDictionary()
    while (cap.isOpened() and a!=ord('q')):
        analyzer.detectCircle()
        analyzer.drawTextInImageText()
        cv2.imshow('Plansza', analyzer.drawBoard()[0])
        cv2.imshow('Zaznaczone pionki', analyzer.drawBoard()[1])
        cv2.imshow('Wizualizacja', analyzer.drawBoard()[2])
        ret, frame = cap.read()
        analyzer.readVideo(frame)
        a = cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()