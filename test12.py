import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import time

img = cv.imread("1_2.jpg", cv.IMREAD_UNCHANGED)

# cv.imshow("img", img)
# cv.imshow("COLORMAP_JET", cv.applyColorMap(img, cv.COLORMAP_JET))
# cv.imshow("COLORMAP_AUTUMN", cv.applyColorMap(img, cv.COLORMAP_AUTUMN))
# cv.imshow("COLORMAP_COOL", cv.applyColorMap(img, cv.COLORMAP_COOL))
# cv.imshow("COLORMAP_CIVIDIS", cv.applyColorMap(img, cv.COLORMAP_CIVIDIS))
# cv.imshow("COLORMAP_PINK", cv.applyColorMap(img, cv.COLORMAP_PINK))
# cv.imshow("COLORMAP_PLASMA", cv.applyColorMap(img, cv.COLORMAP_PLASMA))
# cv.imshow("COLORMAP_WINTER", cv.applyColorMap(img, cv.COLORMAP_WINTER))
# cv.imshow("COLORMAP_SUMMER", cv.applyColorMap(img, cv.COLORMAP_SUMMER))
# cv.imshow("COLORMAP_HOT", cv.applyColorMap(img, cv.COLORMAP_HOT))
# cv.imshow("COLORMAP_HSV", cv.applyColorMap(img, cv.COLORMAP_HSV))
# cv.imshow("COLORMAP_RAINBOW", cv.applyColorMap(img, cv.COLORMAP_RAINBOW))
# cv.imshow("COLORMAP_BONE", cv.applyColorMap(img, cv.COLORMAP_BONE))
# cv.imshow("COLORMAP_OCEAN", cv.applyColorMap(img, cv.COLORMAP_OCEAN))

img1 = cv.cvtColor(img, cv.COLOR_BGR2BGRA)

b, g, r = cv.split(img)

print(b.shape)

alpha = np.ones((512, 512, 3), dtype=np.uint8) * 2

# print("alpha ", alpha.shape)

img2 = cv.merge((b, g, r, alpha))

img1[:, :, 3] = 50

img3 = np.dstack((img, alpha))

img4 = img.copy()

img4 = img4.astype(float)

# print("img4 ", img4.shape)

alpha = alpha.astype(float) / 255

img5 = cv.multiply(alpha, img4)

img6 = cv.GaussianBlur(img5, (5, 5,), 0)

black = np.ones((512, 512, 3), img6.dtype)

img7 = cv.addWeighted(img6, 0.6, black, 0.1, 0)

white_img = np.ones((512, 512, 3), img.dtype) + 50

img8 = cv.addWeighted(img, 0.6, white_img, 0.6, 0)

img9 = cv.subtract(img5, img7)

start_time = time.time()

# for rows in range(len(img2)):
#     for cols in range(len((img2[rows]))):
#         pixel = img2[rows][cols]
#         pixel[3] = 0.0
#         img2[rows][cols] = pixel

print("timing ", (time.time() - start_time))



# cv.imshow("img1", img1)
# cv.imshow("img2", img2)
# cv.imshow("img3", img3)
# cv.imshow("img5", img5)
# cv.imshow("img6", img6)
# cv.imshow("img7", img7)
# cv.imshow("img8", img8)
# cv.imshow("img9", img9)
cv.waitKey(0)
cv.destroyAllWindows()