# -*- coding=utf-8 -*-

import cv2
import numpy as np
import time
import sys


def drawMatches(img1, kp1, img2, kp2, matches):
    """

    @author User rairyeng on StackOVerflow: http://stackoverflow.com/questions/20259025/module-object-has-no-attribute-drawmatches-opencv-python

    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated
    keypoints, as well as a list of DMatch data structure (matches)
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage

    print(img2)


    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]


    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    # out[:rows1,:cols1] = np.dstack([img1, img1, 3])

    # Place the next image to the right of it
    # out[:rows2,cols1:] = np.dstack([img2, img2, 3])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    # Show the image
    # cv2.imshow('Matched Features', img2)
    # cv2.waitKey(0)
    # cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return img2

if __name__ == "__main__":
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = "./sangue-de-boi.jpg"

    src = cv2.imread(fn)
    dst = cv2.Canny(src, 50, 200) # aplica o detector de bordas de Canny à imagem src
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR) # Converte a imagem para BGR para permitir desenho colorido
    video = cv2.VideoCapture("sangue-de-boi.mp4")

    # Valores para o estudo com Homography
    MIN_MATCH_COUNT = 10
    sift = cv2.SIFT()

    while(True):
        # Tira foto (dentro de loop, ou seja, fazemos
        # espécie de slide show)
        image = video.read()

        # Confere se foi possível tirar a foto
        if True:
            cv2.imshow('Webcam', image)
            kp2, des2 = sift.detectAndCompute(image, None)

            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks = 50)

            # Configura o algoritmo de casamento de features
            flann = cv2.FlannBasedMatcher(index_params, search_params)

            # Tenta fazer a melhor comparacao usando o algoritmo
            matches = flann.knnMatch(des1,des2,k=2)

            good = []
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append(m)


            if len(good)>MIN_MATCH_COUNT:
                # Separa os bons matches na origem e no destino
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)


                # Tenta achar uma trasformacao composta de rotacao, translacao e escala que situe uma imagem na outra
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                matchesMask = mask.ravel().tolist()

                h,w = pic.shape[0], pic.shape[1]
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

                # Transforma os pontos da imagem origem para onde estao na imagem destino
                dst = cv2.perspectiveTransform(pts,M)

                # Desenha as linhas
                img2b = cv2.polylines(image,[np.int32(dst)],True,255,3, cv2.CV_AA)

            else:
                print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
                matchesMask = None

            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                               singlePointColor = None,
                               matchesMask = matchesMask, # draw only inliers
                               flags = 2)

            img3 = drawMatches(pic, kp1, image, kp2, good[:20])
            cv2.imshow("final", img3)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            print "Não foi possível localizar sua webcam e fazer um belo video"


    webcam.release()
    cv2.destroyAllWindows()
