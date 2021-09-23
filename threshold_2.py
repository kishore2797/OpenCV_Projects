import cv2 as cv
import numpy as np

img = cv.imread("1_2.jpg")
img1 = cv.imread("2_2.jpg")

img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

img3 = cv.cvtColor(img, cv.COLOR_BGR2RGB)

kernel = np.ones((5, 5), np.float32) / 25

# img4 = cv.filter2D(img, -1, kernel)

img5 = cv.bitwise_not(img3)

img6 = cv.cvtColor(img5, cv.COLOR_RGB2GRAY)

img7 = cv.cvtColor(img6, cv.COLOR_GRAY2BGR)

hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

lower_bound = np.array([130, 50, 50], np.uint8)
upper_bound = np.array([110, 255, 255], np.uint8)

mask = cv.inRange(hsv_img, lower_bound, upper_bound)

img8 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)

res = cv.bitwise_or(img, img7)

img9 = np.concatenate((img, img7), axis=1)

cv.imshow("img", img)
cv.imshow("img1", img1)
cv.imshow("img2", img2)
cv.imshow("img3", img3)
# cv.imshow("img4", img4)
# cv.imshow("img5", img5)
# cv.imshow("img6", img6)
# cv.imshow("img7", img7)
# cv.imshow("res", res)
# cv.imshow("img7_not", cv.bitwise_not(img7))
cv.imshow("img9", img9)
cv.waitKey(0)
cv.destroyAllWindows()
