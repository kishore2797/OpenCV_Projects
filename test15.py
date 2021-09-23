import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import PIL.Image
from sklearn.cluster import KMeans
import random

sam_img = cv.imread("1_2.jpg", cv.IMREAD_UNCHANGED)
sam_img1 = cv.imread("2_2.jpg", cv.IMREAD_UNCHANGED)


# def findNearest_list(rgb):
#     dist = [(((rgbValues[i][0]-rgb[0])*0.3)**2 + ((rgbValues[i][1]-rgb[1])*0.59)**2 + ((rgbValues[i][2]-rgb[2])*0.11)**2,i) for i in range(22)]
#     return rgbValues[min(dist)[1]]


pixels = sam_img.copy()
width, height = sam_img.shape[:2]

# rgbValues = [tuple(random.randrange(0,256) for _ in range(3)) for _ in range(22)]

start_time = time.time()

# for y in range(height):
#     for x in range(width):
#         # fetch the rgb value
#         color = pixels[x,y]
#         # replace with nearest
#         pixels[x,y] = findNearest_list (color)

blueX, greenX, redX = cv.split(sam_img)
blueY, greenY, redY = cv.split(sam_img1)

print(blueX)
print(sam_img)


# from sklearn.neighbors import KNeighborsClassifier
#
# blueClassifier = KNeighborsClassifier(n_neighbors=3)
# greenClassifier = KNeighborsClassifier(n_neighbors=3)
# redClassifier = KNeighborsClassifier(n_neighbors=3)
#
# blueClassifier.fit(blueX, blueY)
# greenClassifier.fit(greenX, greenY)
# redClassifier.fit(redX, redY)

print(time.time() - start_time)

# cv.imshow("pixels", pixels)
# cv.waitKey(0)
# cv.destroyAllWindows()