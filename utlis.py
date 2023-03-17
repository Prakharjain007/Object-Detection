import cv2
import numpy as np

def getContour(img,cThreshold=[100,100],showCanny=False,minArea=1000,filter=0):
    imgGray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur=cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny=cv2.Canny(imgBlur,cThreshold[0],cThreshold[1])
    kenrel= np.ones((5,5))
    imgDial= cv2.dilate(imgCanny,kenrel,iterations=3)
    imgthre=cv2.erode(imgDial,kenrel,iterations=2)
    if showCanny:cv2.imshow("Canny",imgthre)
    contours,hierarachy = cv2.findContours(imgthre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    finalCountours = []

    for i in contours:
        area= cv2.contourArea(i)
        if area > minArea:
            peri = cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,0.02*peri,True)
            bbox = cv2.boundingRect(approx)
            if filter>0:
                if len(approx) == filter:
                    finalCountours.append(len(approx),area,approx,bbox,i)
            else:
                finalCountours.append(len(approx),area,approx,bbox,i)
