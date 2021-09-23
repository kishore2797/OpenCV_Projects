import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("1_2.jpg")
img1 = cv.imread("2_2.jpg")

b, g, r = cv.split(img)
b1, g1, r1 = cv.split(img1)

plt.hist([b.ravel(), g.ravel(), r.ravel(), b1.ravel(), g1.ravel(), r1.ravel()], 256, [0, 256], histtype="step",
         label=["b", "g", "r", "b1", "g1", "r1"])
plt.legend()
plt.show()

# cv.imshow("img1", img1)
# cv.imshow("img3", img3)
# cv.waitKey(0)
# cv.destroyAllWindows()
