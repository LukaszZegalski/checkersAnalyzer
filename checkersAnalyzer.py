import cv2
import numpy as np
from sklearn import cluster
import pyautogui as p


class checkersAnalyzer(object):

    #zmienne klasowe
    def __init__(self, debug):
        #obraz szachownicy z kamery
        self.image = None
        self.debug = debug

        #wspołrzędne do przekształcenia
        self.points = []
        #szerokość obrazu do obróbki
        self.width = 450
        #wysokość obrazu do obróbki
        self.height = 450
        #macierz pozycji pionków
        self.matrix = np.zeros((8, 8))
        #plansza do wizualizacji
        self.board = None
        self.board_2 = None
        self.image_circle = None
        #słownik wspołrzędne w macierzy, współrzędne na obrazie
        self.coordinates = {}
        #wielkość pojedynczego pola w planszy do wizualizacji
        self.height_width_one_area_in_board = 543 // 8


    def readVideo(self,img):
        self.image=img

    #wczytanie zdjecia
    def read(self, path1):
        self.image = cv2.imread(path1, cv2.IMREAD_COLOR)
        self.image = cv2.resize(self.image, (self.width,self.height))

        if self.debug:
            cv2.imshow('Orginal', self.image)

    #pobranie pozycji szachownicy

    def checkboardTransposition(self):
        p = self.points
        pts1 = np.float32([[p[0][0], p[0][1]], [p[1][0], p[1][1]], [p[2][0], p[2][1]], [p[3][0], p[3][1]]])
        pts2 = np.float32([[0, 0], [self.width, 0], [self.width, self.height], [0, self.height]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(self.image, M, (self.width, self.height))
        self.image = dst

    #Przekształcenie pobranie położenia szachownicy
    def checkboardCoordinate(self):
        def onMouse(event, x, y,flaga,a):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.points.append([x,y])
        cv2.imshow('Wejściowy obraz', self.image)
        cv2.setMouseCallback('Wejściowy obraz', onMouse)
        while len(self.points)<4:
            cv2.waitKey(1)

    # Wypełnienie słwnika odpowiednimi wartościami
    def intoDictionary(self):
        for x in range(0, 8):
            for y in range(0, 8):
                self.coordinates.update({str(x) + str(y): [self.height_width_one_area_in_board * (x + 1), self.height_width_one_area_in_board * (y + 1)]})

    #wykrywanie pionków
    def detectCircle(self):
        # zamiana BGR na skalę szarości
        img = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        self.image_circle = self.image.copy()

        # wyliczenie szerkości pól na obrazie z kamery
        width_one_area_image = img.shape[0] / 8
        height_one_area_image = img.shape[1] / 8

        # Wyszukanie kół na obrazie
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 2, 40, param1=100, param2=30, minRadius=18, maxRadius=22)
        circles = np.uint16(np.around(circles))

        # Zaznaczenie środków znalezionych kół oraz ich obrysów
        for i in circles[0, :]:
            cv2.circle(self.image_circle, (i[0], i[1]), i[2], (0, 255, 0), 1);
            cv2.circle(self.image_circle, (i[0], i[1]), 2, (0, 0, 255), 3);

        # Stworzenie tablicy przechowującej wspołrzęnde poszczególnych pionów - tablica 8 na 8
        Matrix2 = []

        # Stworzenie tablicy przechowującej wycinki z centrum poszczególnych pionków
        center_circle = []

        # Wypełnienie tablicy Matrix2 oraz centra
        for x in circles[0]:
            x1 = int(x[1] // width_one_area_image)
            y1 = int(x[0] // height_one_area_image)
            cut_area = img[x[1] - 5:x[1] + 5, x[0] - 5:x[0] + 5]
            ret, thresh1 = cv2.threshold(cut_area, 160, 255, cv2.THRESH_BINARY)
            center_circle.append(thresh1.copy())
            Matrix2.append([y1, x1])

        # stworzenie pomocniczych tablic do spłaszczenia
        pom = []
        centra_final = []

        # Spłaszczenie tablicy do postaci [ [piksel, piksel, ....] , [] , [] , .....]
        for x in center_circle:
            for y in x:
                for z in y:
                    pom.append(z)
            centra_final.append(pom.copy())
            pom.clear()

        # Uzupełnianie tablicy Matrix odpowiednimi pionami na danej pozycji oraz narysowanie ich na Plansza
        self.board=self.board2.copy()
        for x, y in zip(Matrix2, center_circle):
            if y[4,4]==0:
                self.matrix[x[1], x[0]] = 1
                cv2.circle(self.board,
                           (self.coordinates[str(x[1]) + str(x[0])][1] - 34, self.coordinates[str(x[1]) + str(x[0])][0] - 34), 20,
                           (0, 0, 255), -1)
            else:
                self.matrix[x[1], x[0]] = 2
                cv2.circle(self.board,
                           (self.coordinates[str(x[1]) + str(x[0])][1] - 34, self.coordinates[str(x[1]) + str(x[0])][0] - 34), 20,
                           (0, 255, 0), -1)

    def drawBoard(self):
        return self.image_circle, self.image, self.board
        #cv2.imshow('Zaznaczone pionki', self.image_circle)
        #cv2.imshow('Plansza', self.image)
        #cv2.imshow('Wizualizacja', self.board)


    def detectAreaBoardDistribution(self):
        # zamiana BGR na skalę szarości
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # wyliczenie szerkości pól na obrazie z kamery
        width_one_area_image = img.shape[0] // 8
        height_one_area_image = img.shape[1] // 8

        cut_area = img[width_one_area_image*3:width_one_area_image*4,height_one_area_image:width_one_area_image*2]
        ret, thresh1 = cv2.threshold(cut_area, 175, 255, cv2.THRESH_BINARY)

        if thresh1[width_one_area_image//2,height_one_area_image//2]==0:
            self.board = cv2.imread('./Picture/board_r.jpg', cv2.IMREAD_COLOR)
            self.board2 = cv2.imread('./Picture/board_r.jpg', cv2.IMREAD_COLOR)
        else:
            self.board = cv2.imread('./Picture/board_l.jpg', cv2.IMREAD_COLOR)
            self.board2 = cv2.imread('./Picture/board_l.jpg', cv2.IMREAD_COLOR)

    def analyzeCircleonBoard(self):
        white = []
        black = []
        pom = 0
        for x in self.matrix:
            for y in x:
                if pom%2 == 0:
                    white.append(y)
                else:
                    black.append(y)
                pom+=1
            pom = 0
        print(white)
        print(black)

    def getPulpit(self):
        img = p.screenshot()
        img = np.array(img)
        self.image = img

