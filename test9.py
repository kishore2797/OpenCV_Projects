import cv2 as cv
import numpy as np
import math

img = cv.imread("1_2.jpg")

height, width = 100, 100

img1 = list(range(width * height))

oilRange = 7

rHis = list(range(256))
gHis = list(range(256))
bHis = list(range(256))

bytesInARow = math.ceil(width * 3 // 4) * 4

for y in range(height):
    for x in range(width):
        for row in range(-oilRange, oilRange):
            rowOffset = y + row
            if 0 <= rowOffset < height:
                for col in range(-oilRange, oilRange):
                    colOffset = x + col
                    if 0 <= colOffset < width:
                        print(rowOffset*width+ colOffset)
        #                 r, g, b = img1[rowOffset*bytesInARow, colOffset]
        #
        #                 rHis[r]+=1
        #                 gHis[g]+=1
        #                 bHis[b]+=1
        #
        # maxR, maxG, maxB = 0, 0, 0
        #
        # for i in range(1, 256):
        #     if rHis[i] > rHis[maxR]:
        #         maxR = i
        #     if gHis[i] > gHis[maxG]:
        #         maxG = i
        #     if bHis[i] > bHis[maxB]:
        #         maxB = i
        #
        # if rHis[maxR] != 0 and gHis[maxG] != 0 and bHis[maxB] != 0:
        #     finalR = maxR
        #     finalG = maxG
        #     finalB = maxB
        #
        #     finalR = min(255, max(0, finalR))
        #     finalG = min(255, max(0, finalG))
        #     finalB = min(255, max(0, finalB))
        #
        #     img1[x*3, y*bytesInARow] = [finalR, finalB, finalG]

