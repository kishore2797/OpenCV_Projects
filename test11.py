import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import time

img = cv.imread("1_2.jpg")

# start_time = time.time()
#
# for rows in range(len(img)):
#     for cols in range(len((img[rows]))):
#         print(img[rows][cols])
#
# print(start_time)

start_time = time.time()

b, g, r = cv.split(img)

print(img[0][0])
print(img[0][1])
print(b[0][0])
print(g[0][0])
print(r[0][0])

img1 = np.array([b,g,r], dtype=np.uint8)

print(img1.shape)

img1_sum = np.sum(img1, axis=0)

print(img1_sum)

print(time.time() - start_time)