import numpy as np
import cv2 as cv
from cv2 import VideoCapture
from cv2 import waitKey
import utlis

##########################################
webcam=False
path = "1.jpg"
cap = cv.VideoCapture(0)
cap.set(10,160)
cap.set(3,1920)
cap.set(4,1080)

while True:
    if webcam:
        success,img=cap.read()
    else:
        img=cv.imread(path)
    
    utlis.getContour(img,showCanny=True)
    img=cv.resize(img,(0,0),None,0.5,0.5)
    cv.imshow("Orignal",img)
    cv.waitKey(0)

