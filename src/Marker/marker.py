import cv2
import numpy as np
from matplotlib import pyplot as plt
import math


global xcenterm1, ycenterm1, xcenterm2, ycenterm2
global xm1lo, ym1lo, xm1ro, ym1ro, xm1ru,ym1ru, xm1lu, ym1lu, xm2lo, ym2lo, xm2ro, ym2ro, xm2ru,ym2ru, xm2lu, ym2lu
global font

def getmarker(x):
    global img
    img = cv2.imread(x ,0)
    
    imgcolor = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    retval	 =	cv2.aruco.getPredefinedDictionary(	cv2.aruco.DICT_ARUCO_ORIGINAL)
    
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(imgcolor, retval)
    
    image = cv2.aruco.drawDetectedMarkers(imgcolor, corners, ids, (0,250,0)) 
    print (corners)
    if len(corners) == 0:
        print ('no markers found!')
        cv2.putText(image, 'no markers found!', (100,100), cv2.FONT_HERSHEY_TRIPLEX, 1, 250)
    cv2.imshow('marker', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
getmarker(r'C:\path')

def getmarkercoordinates(x):
    img = cv2.imread(x ,0)
    
    imgcolor = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    retval	 =	cv2.aruco.getPredefinedDictionary(	cv2.aruco.DICT_ARUCO_ORIGINAL)
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(imgcolor, retval) 
  
    global xm1lo, ym1lo, xm1ro, ym1ro, xm1ru,ym1ru, xm1lu, ym1lu, xm2lo, ym2lo, xm2ro, ym2ro, xm2ru,ym2ru, xm2lu, ym2lu,  xm3lo, ym3lo, xm3ro, ym3ro, xm3ru,ym3ru, xm3lu, ym3lu
    if len(corners) > 2:
        xm1lo = corners[0][0][0][0]
        ym1lo = corners[0][0][0][1]
        xm1ro = corners[0][0][1][0]
        ym1ro = corners[0][0][1][1]
        xm1ru = corners[0][0][2][0]
        ym1ru = corners[0][0][2][1]
        xm1lu = corners[0][0][3][0]
        ym1lu = corners[0][0][3][1]
        
        xm2lo = corners[1][0][0][0]
        ym2lo = corners[1][0][0][1]
        xm2ro = corners[1][0][1][0]
        ym2ro = corners[1][0][1][1]
        xm2ru = corners[1][0][2][0]
        ym2ru = corners[1][0][2][1]
        xm2lu = corners[1][0][3][0]
        ym2lu = corners[1][0][3][1]
        
        xm3lo = corners[2][0][0][0]
        ym3lo = corners[2][0][0][1]
        xm3ro = corners[2][0][1][0]
        ym3ro = corners[2][0][1][1]
        xm3ru = corners[2][0][2][0]
        ym3ru = corners[2][0][2][1]
        xm3lu = corners[2][0][3][0]
        ym3lu = corners[2][0][3][1]
        
        m1 = np.array([[xm1lo, ym1lo], [xm1ro, ym1ro], [xm1ru,ym1ru], [xm1lu, ym1lu]])
        m2 = np.array([[xm2lo, ym2lo], [xm2ro, ym2ro], [xm2ru,ym2ru], [xm2lu, ym2lu]])
        m3 = np.array([[xm3lo, ym3lo], [xm3ro, ym3ro], [xm3ru,ym3ru], [xm3lu, ym3lu]])
        
        print ('Koordinaten Marker1:', m1)
        print ('Koordinaten Marker 2:', m2)
        print ('Koordinaten Marker 3:', m3)

    
#Center marker 1, 2, 3
        global xcenterm1, ycenterm1, xcenterm2, ycenterm2, xcenterm3, ycenterm3
        xcenterm1 = xm1lo + ((xm1ro - xm1lo)/2)
        ycenterm1 = ym1ro + ((ym1ru - ym1ro)/2)
        xcenterm2 = xm2lo + ((xm2ro - xm2lo)/2)
        ycenterm2 = ym2ro + ((ym2ru - ym2ro)/2)
        xcenterm3 = xm3lo + ((xm3ro - xm3lo)/2)
        ycenterm3 = ym3ro + ((ym3ru - ym3ro)/2)

        
        print ('Koordinaten center Marker1:', int(xcenterm1), int(ycenterm1))
        print ('Koordinaten center Marker2:', int(xcenterm2), int(ycenterm2))
        print ('Koordinaten center Marker3:', int(xcenterm3), int(ycenterm3))
    else:
        print ('no three markers found')
    
getmarkercoordinates(r'C:\path')

def getdistance(x):
    font = cv2.FONT_HERSHEY_TRIPLEX
    img = cv2.imread(x ,0)
    
    imgcolor = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    retval	 =	cv2.aruco.getPredefinedDictionary(	cv2.aruco.DICT_ARUCO_ORIGINAL)
    
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(imgcolor, retval)
    image = cv2.aruco.drawDetectedMarkers(imgcolor, corners, ids, (0,250,0)) 
    
    if len(corners) != 0 and len(corners) !=1:
        
        distance1 = math.sqrt(((ycenterm2-ycenterm1)**2)+((xcenterm2-xcenterm1)**2))
        distance2 = math.sqrt(((ycenterm3-ycenterm1)**2)+((xcenterm3-xcenterm1)**2))
        distance1int = int(distance1)
        distance2int = int(distance2)
        print (distance1int, distance2int)
        print ("Distance is:", distance1int, 'and', distance2int)
        
        cv2.line(image, (int(xcenterm1), int(ycenterm1)), (int(xcenterm2), int(ycenterm2)), (0, 250,0), thickness=3, lineType=5, shift=0)
        cv2.line(image, (int(xcenterm1), int(ycenterm1)), (int(xcenterm3), int(ycenterm3)), (0, 250,0), thickness=3, lineType=5, shift=0)
        
        xmittedist1 = xcenterm1 + ((xcenterm2-xcenterm1)/2)
        ymittedist1 = ycenterm1 + ((ycenterm2-ycenterm1)/2)
        xmittedist2 = xcenterm1 + ((xcenterm3-xcenterm1)/2)
        ymittedist2 = ycenterm1 + ((ycenterm3-ycenterm1)/2)  
        
        cv2.putText(image, str(distance1int), (int(xmittedist1), int(ymittedist1)), font, 1, 250)
        cv2.putText(image, str(distance2int), (int(xmittedist2), int(ymittedist2)), font, 1, 250)
        cv2.imshow('test', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print ('nothing found')
    
getdistance(r'C:\path')
