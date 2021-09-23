import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import PIL.Image
from sklearn.cluster import KMeans

sam_img = cv.imread("1_2.jpg", cv.IMREAD_UNCHANGED)


def limit_size(img, max_x, max_y=0):

    if max_x == 0:
        return img

    if max_y == 0:
        max_y = max_x

    ratio = min(1.0, float(max_x) / img.shape[1], float(max_y) / img.shape[0])

    if ratio != 1.0:
        shape = (int(img.shape[1] * ratio), int(img.shape[0] * ratio))
        return cv.resize(img, shape, interpolation=cv.INTER_AREA)
    else:
        return img


start_time = time.time()

sam_img1 = limit_size(sam_img, 300)

# cv.imshow("sam_img1", sam_img1)
# cv.waitKey(0)
# cv.destroyAllWindows()

clusters = 4

clt = KMeans(n_clusters=clusters)
clt.fit(sam_img1.reshape(-1, 3))

colors = clt.cluster_centers_

# print(colors)

# cols = len(colors)
# rows = int(math.ceil(cols / cols))
#
# res = np.zeros((rows * 80, cols * 80, 3), dtype=np.uint8)

print(colors.shape)

new_colors = np.array(colors, dtype=np.uint8)

# print(new_colors)
#
# print(np.sum(new_colors))

result = np.split(new_colors, indices_or_sections=[1,2], axis=1)

# print(result)
#
# print(np.sum(result, axis=0))
#
# print(sam_img[30:50, 30:50])

zero_pad = 256 * 3 - len(colors)

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


img = createLUT(80)

print(img.shape)

print(len(img))

# print(img[0])

blue, green, red = cv.split(sam_img1)

palette = result.copy()

# print(len(palette))

base_len = len(colors)

ratio = 256 / base_len

# print(result[0])


def createTable(cust_colors, color_len):
    # print(cust_colors)

    loopUpTable = np.zeros((1, 256), dtype=np.uint8)

    x = 0

    startIdx = 0

    paletteIdx = -1

    while x <= 256:
        for y in range(startIdx, x):
            # if y > 224:
            #     print("cust_colors[paletteIdx] ", cust_colors[paletteIdx])
            for z in range(len(loopUpTable)):
                loopUpTable[z, y] = cust_colors[paletteIdx]

        startIdx = x

        # print("startIdx ", startIdx)

        paletteIdx += 1

        # print("paletteIdx ", paletteIdx)

        x += 256 // color_len

        # print("x ", x)

    return loopUpTable


blueLUT = createTable(result[0], clusters)
greenLUT = createTable(result[1], clusters)
redLUT = createTable(result[2], clusters)

# print(blueLUT)
# print(greenLUT)
# print(redLUT)

cv.LUT(blue, blueLUT, blue)
cv.LUT(green, greenLUT, green)
cv.LUT(red, redLUT, red)

sam_img2 = cv.merge((blue, green, red))

cv.imwrite("sam_img2.jpg", sam_img2)

result_LUT = cv.merge((blueLUT, greenLUT, redLUT))

print(result_LUT.shape)

from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3)




print("time ", (time.time() - start_time))

cv.imshow("sam_img2", sam_img2)
cv.imshow("result_LUT", result_LUT)
cv.waitKey(0)
cv.destroyAllWindows()




