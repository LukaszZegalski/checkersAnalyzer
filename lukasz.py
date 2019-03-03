from pylab import *
import cv2
import numpy as np
from sklearn import cluster

#tablica reprezentująca szachownicę jako macierz
Matrix = np.zeros((8,8))

#Plansza do wizualizacji
Plansza = cv2.imread('plansza2.jpg',cv2.IMREAD_COLOR)

#Słownik klucz: współrzędne tablicowe, wartość: współrzędne na obrazie
wspolrzedne = {}

#Szerokość, Wysokość pojedynczego pola w plamszy
X_plansza = 543//8
Y_plansza = 544//8

#Wypełnienie słwnika odpowiednimi wartościami
for x in range(0,8):
    for y in range(0,8):
        wspolrzedne.update({str(x)+str(y):[X_plansza*(x+1),Y_plansza*(y+1)]})

#Wczytanie zdjęcia z warcabami, przeskalowanie, zamiana na skalę szarości
img = cv2.resize(cv2.imread('checkers.png',cv2.IMREAD_GRAYSCALE),(450,454))
color = cv2.resize(cv2.imread('checkers.png',cv2.IMREAD_COLOR),(450,454))

#wyliczenie szerkości pól
X_wczytane=img.shape[0]/8
Y_wczytane=img.shape[1]/8

#zamiana na RGB w celu zaznaczenia położenia pionków
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

#Wyszukanie kół na obrazie
circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,2,40,param1=100,param2=30,minRadius=18,maxRadius=22)
circles = np.uint16(np.around(circles))

#Zaznaczenie środków znalezionych kół oraz ich obrysów
for i in circles[0,:]:
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),1);
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3);

#Stworzenie tablicy przechowującej wspołrzęnde poszczególnych pionów - tablica 8 na 8
Matrix2 = []

#Stworzenie tablicy przechowującej wycinki z centrum poszczególnych pionków
centra=[]

#Wypełnienie tablicy Matrix2 oraz centra
for x in circles[0]:
    x1 = int(x[1]//X_wczytane)
    y1 = int(x[0]//Y_wczytane)
    fragment = img[x[1]-5:x[1]+5,x[0]-5:x[0]+5]
    centra.append(fragment.copy())
    Matrix2.append([y1,x1])

#stworzenie pomocniczych tablic do spłaszczenia
pom =[]
centra_final = []

#Spłaszczenie tablicy do postaci [ [piksel, piksel, ....] , [] , [] , .....]
for x in centra:
    for y in x:
        for z in y:
            pom.append(z)
    centra_final.append(pom.copy())
    pom.clear()

#wykorzystanie k-średnich w celu podziału zbioru na 2 grupy, białe oraz czarne pionki
kernel = np.ones((3, 3), np.uint8)
k_means = cluster.KMeans(n_clusters=2)
k_means.fit(centra_final)
labels = k_means.labels_

#Uzupełnianie tablicy Matrix odpowiednimi pionami na danej pozycji oraz narysowanie ich na Plansza
for x, y in zip(Matrix2, labels):
    if y==0:
        Matrix[x[1],x[0]]=1
        cv2.circle(Plansza, (wspolrzedne[str(x[1])+str(x[0])][1]-34,wspolrzedne[str(x[1])+str(x[0])][0]-34), 20, (0, 0, 255), -1)
    else:
        Matrix[x[1], x[0]] = 2
        cv2.circle(Plansza, (wspolrzedne[str(x[1])+str(x[0])][1]-34,wspolrzedne[str(x[1])+str(x[0])][0]-34), 20, (0, 255, 0), -1)

a=ord('a')
while a!=ord('q'):
    cv2.imshow('plansza_zaznaczone',cimg)
    cv2.imshow('plansza',color)
    cv2.imshow('Wizualizacja',Plansza)
    a = cv2.waitKey(1)