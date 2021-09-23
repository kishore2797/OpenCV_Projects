import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import time

sam_img = cv.imread("1_2.jpg", cv.IMREAD_UNCHANGED)


def createLUT(num_colors):
    lookUpTable = np.zeros((1, 256), np.uint8)

    startIdx = 0

    x = 0

    while x < 256:
        lookUpTable[0, x] = x

        for y in range(startIdx, x):
            if lookUpTable[0, y] == 0:
                lookUpTable[0, y] = lookUpTable[0, x]

        startIdx = x

        x += 256 // num_colors

    return lookUpTable


# color reduction using LUT for gray images

def reduceColorsGray(img, numColors):
    lookUpTable = createLUT(numColors)

    return cv.LUT(img, lookUpTable)


sam_img1 = cv.cvtColor(sam_img, cv.COLOR_BGR2GRAY)

# use 5 colors
sam_img2 = reduceColorsGray(sam_img1, numColors=5)

# use 2 colors
sam_img3 = reduceColorsGray(sam_img1, numColors=2)

# cv.imshow("sam_img2", sam_img2)
# cv.imshow("sam_img3", sam_img3)
# cv.waitKey(0)
# cv.destroyAllWindows()

# by increasing the numColors, the LUT approaches the gray scale image
# if numColors is 256, reduceColorsGray function returns the full gray scale

# color reduction using LUT for gray images.

# color image actually consists of a no.of gray images.
# for applying LUT over color images, there are 2 techniques.
# 1. Using a single LUT for all color image channels
# 2. Using a single LUT for each color channels
# lets try these 2 approaches

# 1. Using a single LUT for all color image channels

numColors = 10

lookUpTable = createLUT(numColors)

proc_img = cv.LUT(sam_img, lookUpTable)

# cv.imshow("proc_img", proc_img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# if the no.of colors increases, the image tends to be more similar to the original color image.

# 2. Using a single LUT for each color channel


def reduceColors(img, numBlue, numGreen, numRed):
    blue, green, red = cv.split(img)

    blueLUT = createLUT(numBlue)

    greenLUT = createLUT(numGreen)

    redLUT = createLUT(numRed)

    cv.LUT(blue, blueLUT, blue)
    cv.LUT(green, greenLUT, green)
    cv.LUT(red, redLUT, red)

    return cv.merge((blue, green, red))


img = reduceColors(sam_img, 50, 1, 50)

# cv.imshow("img", img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# now use adaptive threshold

img_gray = cv.cvtColor(sam_img, cv.COLOR_BGR2GRAY)

img_gray_1 = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, 2)

# remove noise from the img_gray for more filter

# use median filter for smoothing the image

cv.medianBlur(img_gray, 5, img_gray)

cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, 2, img_gray_1)

reducedColorImage = reduceColors(sam_img, 80, 100, 10)

# convert img_gray to color

img_color = cv.cvtColor(img_gray_1, cv.COLOR_GRAY2BGR)

result = cv.bitwise_and(reducedColorImage, img_color)

plt.imshow(cv.cvtColor(reducedColorImage, cv.COLOR_BGR2RGB), 'gray')
plt.show()

cv.imshow("img_gray", img_gray)
cv.imshow("img_gray_1", img_gray_1)
cv.imshow("img_color", img_color)
cv.imshow("reducedColorImage", reducedColorImage)
cv.imshow("result", result)
cv.waitKey(0)
cv.destroyAllWindows()