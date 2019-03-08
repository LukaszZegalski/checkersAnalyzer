import cv2
import numpy as np
import pyautogui as p
import collections as co

class checkersAnalyzer(object):

    #zmienne klasowe
    def __init__(self, debug):
        self.debug = debug
        self.image = None #obraz szachownicy z kamery
        self.points = [] #wspołrzędne do przekształcenia
        self.wh_size= 450 #szerokość, wysokość obrazu do obróbki
        self.matrix_pawns = np.zeros((8, 8)) #macierz pozycji pionków aktualna
        self.matrix_old_pawns = np.zeros((8, 8)) #macierz pozycji pionków poprzednia
        self.coordinates = {} #słownik wspołrzędne w macierzy, współrzędne na obrazie
        self.sq = 68 #wielkość pojedynczego pola w planszy do wizualizacji
        self.checkers_size = 544 #rozmiar planszy do wizualizacji
        self.board = None #Wizualizacja planszy
        self.first_sq = None


    #ustawienie klatki z filmu/kamery jako img
    def readVideo(self,img):
        self.image=cv2.resize(img,(self.wh_size,self.wh_size))

    #wczytanie zdjecia
    def read(self, path1):
        self.image = cv2.imread(path1, cv2.IMREAD_COLOR)
        self.image = cv2.resize(self.image, (self.wh_size,self.wh_size))

    #transformacja morfologiczna wyodrębniająca szachownicę na podstawie wczęsniej podanych parametrów
    def checkboardTransposition(self):
        p = self.points
        pts1 = np.float32([[p[0][0], p[0][1]], [p[1][0], p[1][1]], [p[2][0], p[2][1]], [p[3][0], p[3][1]]])
        pts2 = np.float32([[0, 0], [self.wh_size, 0], [self.wh_size, self.wh_size], [0, self.wh_size]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(self.image, M, (self.wh_size, self.wh_size))
        self.image = dst

    # pobranie pozycji szachownicy w przypadku odręcznego zaznaczania jej położenia
    def checkboardCoordinate(self):
        def onMouse(event, x, y,flaga,a):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.points.append([x,y])
        cv2.imshow('Wejściowy obraz', self.image)
        cv2.setMouseCallback('Wejściowy obraz', onMouse)
        while len(self.points)<4:
            cv2.waitKey(1)

    #Wypełnienie słwnika odpowiednimi wartościami (x,y) -> pozycja tablicy : [a,b] -> pozycja pikseli w wizualizacji
    def intoDictionary(self):
        for x in range(0, 8):
            for y in range(0, 8):
                self.coordinates.update({str(x) + str(y): [self.sq * (x + 1), self.sq * (y + 1)]})

    #tworzenie szachownucy position=0 -> lewy grny czarny, position=1 -> lewy górny biały
    def createCheckers(self):
        img = np.zeros((self.checkers_size,self.checkers_size, 3), dtype=np.uint8)
        c = np.fromfunction(lambda x, y: ((x // self.sq) + (y // self.sq)) % 2, (self.checkers_size, self.checkers_size))
        if self.first_sq == 1:
            img[c == 0] = (0,0,0)
            img[c == 1] = (255,255,255)
        else:
            img[c == 0] = (255,255,255)
            img[c == 1] = (0,0,0)
        return img

    #Wypisanie aktualnego stanu pionków
    def drawTextInImageText(self):
        text = np.zeros((136,self.checkers_size, 3), dtype=np.uint8)
        text[::]=(128,128,128)
        ilosc = co.Counter(self.matrix_pawns.flatten())
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(text, 'Biale: '+str(ilosc[2.0]), (150, 60), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(text, 'Czarne: '+str(ilosc[1.0]), (110, 125), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        return text

    #przekazanie parametrów w celu wyświetlenia obrazu, złączenie szachownicy z napisami
    def drawBoard(self):
        new_board = np.concatenate((self.board,self.drawTextInImageText()),axis=0)
        return self.image_circle, self.image, new_board
        cv2.imshow('Zaznaczone pionki', self.image_circle)
        cv2.imshow('Plansza', self.image)
        cv2.imshow('Wizualizacja', new_board)

    #Wykrycie rozkładu pół na szachownicy
    def detectAreaBoardDistribution(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        width = img.shape[0] // 8
        height = img.shape[1] // 8

        cut_area = img[width*3:width*4,height:width*2]
        thresh = cv2.threshold(cut_area, 175, 255, cv2.THRESH_BINARY)[1]

        if thresh[width//2,height//2]==0:
            self.first_sq = 1
        else:
            self.first_sq = 0

    #wykrywanie pionków
    def detectCircle(self):
        # zamiana BGR na skalę szarości
        img = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        self.image_circle = self.image.copy()

        # wyliczenie szerkości pól na obrazie z kamery
        width_sq = img.shape[0] / 8
        height_sq =  img.shape[1] / 8

        # Wyszukanie kół na obrazie
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 2, 40, param1=100, param2=30, minRadius=18, maxRadius=22)
        circles = np.uint16(np.around(circles))

        # Zaznaczenie środków znalezionych kół oraz ich obrysów
        for i in circles[0, :]:
            cv2.circle(self.image_circle, (i[0], i[1]), i[2], (0, 255, 0), 1);
            cv2.circle(self.image_circle, (i[0], i[1]), 2, (0, 0, 255), 3);


        matrix2 = []  # Stworzenie tablicy przechowującej wspołrzęnde poszczególnych pionów - tablica 8 na 8
        center_circle = [] # Stworzenie tablicy przechowującej wycinki z centrum poszczególnych pionków

        # Wypełnienie tablicy Matrix2 oraz centra
        for x in circles[0]:
            x1 = int(x[1] // width_sq)
            y1 = int(x[0] // height_sq)
            cut_area = img[x[1] - 5:x[1] + 5, x[0] - 5:x[0] + 5]
            ret, thresh1 = cv2.threshold(cut_area, 160, 255, cv2.THRESH_BINARY)
            center_circle.append(thresh1.copy())
            matrix2.append([y1, x1])

        # Aktualizacja matrix_pawns oraz rysowania pionów
        self.matrix_old_pawns = self.matrix_pawns.copy()
        self.matrix_pawns[::] = 0
        self.board = self.createCheckers()
        for x, y in zip(matrix2, center_circle):
            if y[4,4]==0:
                self.matrix_pawns[x[1], x[0]] = 1
                cv2.circle(self.board,
                           (self.coordinates[str(x[1]) + str(x[0])][1] - 34, self.coordinates[str(x[1]) + str(x[0])][0] - 34), 20,
                           (64,64,64), -1)
            else:
                self.matrix_pawns[x[1], x[0]] = 2
                cv2.circle(self.board,
                           (self.coordinates[str(x[1]) + str(x[0])][1] - 34, self.coordinates[str(x[1]) + str(x[0])][0] - 34), 20,
                           (217,217,217), -1)
