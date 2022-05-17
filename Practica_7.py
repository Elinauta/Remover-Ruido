from msilib.schema import Binary
import cv2
from cv2 import MORPH_RECT
from cv2 import erode
from cv2 import dilate
from cv2 import blur
import numpy as np

video = cv2.VideoCapture(0)

azulBajo1 = np.array([94,100,20], np.uint8)
azulAlto1 = np.array([126,255,255], np.uint8)

while True:
    ret,frame = video.read()
    if ret==True:
        frame_hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #Se genera una mascara con el rango de color AZUL
        mascara_azul=cv2.inRange(frame_hsv, azulBajo1, azulAlto1)
        #                Detección de color Azul
        #Todo valor que está en el rango de la mascara se muestra 
        solo_azul=cv2.bitwise_and(frame, frame, mask=mascara_azul)
        
        #------------------------LINEALES-------------------------------
        kernel1=np.ones((5,5), np.float32)/25

        filtro1 = cv2.filter2D(mascara_azul,-1,kernel1)
        filtro2 =cv2.blur(mascara_azul,(3,3))
        filtro3=cv2.GaussianBlur(mascara_azul,(5,5),0)
        filtro4=cv2.medianBlur(mascara_azul,9)
        filtro5=cv2.bilateralFilter(mascara_azul,15,75,75)

        #----------------------MORFOLOGICAS-----------------------------
        kernel2 = np.ones((5,5),np.uint8)

        erosion = cv2.erode(mascara_azul,kernel2,iterations = 1)
        dilatacion = cv2.dilate(mascara_azul,kernel2,iterations = 1)
        apertura = cv2.morphologyEx(mascara_azul, cv2.MORPH_OPEN, kernel2)
        cierre = cv2.morphologyEx(mascara_azul, cv2.MORPH_CLOSE, kernel2)
        gradiente = cv2.morphologyEx(mascara_azul, cv2.MORPH_GRADIENT, kernel2)

        cv2.imshow('Frame', frame)

        cv2.imshow('Mascara BINARIA Azul', mascara_azul)
        cv2.imshow('Mascara Azul', solo_azul)

        cv2.imshow('Filtro 1', filtro1)
        cv2.imshow('Filtro 2', filtro2)
        cv2.imshow('Filtro 3', filtro3)
        cv2.imshow('Filtro 4', filtro4)
        cv2.imshow('Filtro 5', filtro5)

        cv2.imshow('Erosion', erosion)
        cv2.imshow('Dilatacion', dilatacion)
        cv2.imshow('Apertura', apertura)
        cv2.imshow('Cierre', cierre)
        cv2.imshow('Gradiente', gradiente)
        
        if cv2.waitKey(1) & 0xFF== ord('s'):
            break

video.release()        
cv2.destroyAllWindows()