import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

img = cv.imread("1_2.jpg")

c = 255 / np.log(1 + np.max(img))

print(c)

log_transformed = c * np.log(1 + img)

print(log_transformed)

log_transformed = np.array(log_transformed, dtype=np.uint8)

white = np.ones((512, 512), dtype=np.uint8) * 255
light = cv.merge((white, white, white))

img1 = cv.addWeighted(img, 1.0, light, 0.2, 0)

img_min = np.min(img)
img_max = np.max(img)

img2 = 256 * (img - img_min) / (img_max - img_min)

img3 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

img3_min = np.min(img3)
img3_max = np.max(img3)

img3 = 256 * (img3 - img3_min) / (img3_max - img3_min)

cv.imshow("log_transformed", log_transformed)
cv.imshow("img1", img1)
cv.imshow("img2", img2)
cv.imshow("img3", img3)
cv.waitKey(0)
cv.destroyAllWindows()
