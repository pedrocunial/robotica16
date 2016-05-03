# -*- coding=utf-8 -*-

import cv2
import numpy as np
import math

cap = cv2.VideoCapture('hall_box_battery.mp4')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    red = ([185, 0, 0], [255, 150, 150])

    lower = np.array(red[0], dtype = "uint8")
    upper = np.array(red[1], dtype = "uint8")

    mask = cv2.inRange(gray, lower, upper)
    output = cv2.bitwise_and(frame, frame, mask = mask)


    red_image = output

    # print(__doc__)


    dst = cv2.Canny(frame, 100, 200) # aplica o detector de bordas de Canny à imagem src
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR) # Converte a imagem para BGR para permitir desenho colorido

    if True: # HoughLinesP
        lines = cv2.HoughLinesP(dst, 1, math.pi/180.0, 40, np.array([]), 50, 10)
        # print("Used Probabilistic Rough Transform")
        # print("The probabilistic hough transform returns the end points of the detected lines")
        a,b,c = lines.shape
        for i in range(a):
            # Faz uma linha ligando o ponto inicial ao ponto final, com a cor vermelha (BGR)
            cv2.line(cdst, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.CV_AA)

    else:    # HoughLines
        # Esperemos nao cair neste caso
        lines = cv2.HoughLines(dst, 1, math.pi/180.0, 50, np.array([]), 0, 0)
        a,b,c = lines.shape
        for i in range(a):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0, y0 = a*rho, b*rhos
            pt1 = ( int(x0+1000*(-b)), int(y0+1000*(a)) )
            pt2 = ( int(x0-1000*(-b)), int(y0-1000*(a)) )
            cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.CV_AA)
        # print("Used old vanilla Hough transform")
        # print("Returned points will be radius and angles")
    # plt.imshow(red_image)

    final = cv2.bitwise_or(red_image, cdst)
    # Display the resulting frame
    cv2.imshow('a-ha - Take On Me (Official Video)',final)
    # cv2.imshow('detected lines', cdst)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
