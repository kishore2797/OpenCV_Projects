import cv2 as cv
import numpy as np
import util

# ksize must be odd.

# img = cv.imread("1.jpg")
img = cv.imread("2.jpeg")

img1 = cv.resize(img, (512, 512))

img2 = util.createAndReturnPencilSketch(img1, (21, 21), 0)
img3 = util.createAndReturnPencilSketchLight(img1, (21, 21), 0)
img4 = util.createAndReturnPencilSketchDark(img1, (21, 21), 0)

img5 = cv.cvtColor(img3, cv.COLOR_GRAY2BGR)

# print(img5)

# cv.imshow("img1", img1)
# cv.imshow("img2", img2)
# cv.imshow("img3", img3)
# cv.imshow("img4", img4)
# cv.imshow("img5", img5)
# cv.waitKey(0)
# cv.destroyAllWindows()


# b, g, r = cv.split(img5)

b, g, r = cv.split(img1)

print(r)
print(r.shape)

# r1 = np.ones((512, 512), dtype=np.uint8) * 255
# g1 = np.ones((512, 512), dtype=np.uint8) * 255

r1 = np.ones((512, 512), dtype=np.uint8)
g1 = np.ones((512, 512), dtype=np.uint8)

print(r1.shape)

# for i in range(len(r1)):
#     for j in range(len(r1[i])):
#         if 220 < r[i][j] < 256:
#             r1[i][j] = 255
#         if 150 < r[i][j] < 221:
#             r1[i][j] = r[i][j] + 35
#         else:
#             r1[i][j] = r[i][j]

img6 = cv.merge((b, g, r1))

# print(img6.shape)

# cv.imshow("img1", img1)
# cv.imshow("img6", img6)
# cv.waitKey(0)
# cv.destroyAllWindows()

img5_hsv = cv.cvtColor(img5, cv.COLOR_BGR2HSV)

img7 = cv.add(img1, img5_hsv)
img8 = cv.add(img1, img5)
img9 = cv.add(img1, img8)

img10 = cv.cvtColor(img4, cv.COLOR_GRAY2BGR)

img11 = cv.add(img8, img10)

img12 = cv.bitwise_and(img1, img7)

img13 = 40 - cv.divide(30 - img9, 20 - img10)

img14 = cv.add(img9, img8)

img15 = cv.subtract(img14, img1)
img16 = cv.subtract(img14, img10)

img17 = cv.addWeighted(img1, 1.0, img14, 0.5, 0)

img18 = cv.addWeighted(img1, 0.8, img8, 0.5, 0)

img19 = cv.subtract(img17, img13)

img20 = cv.addWeighted(img18, 0.6, img19, 0.4, 0)

img21 = cv.imread("1.jpg")

img21 = cv.resize(img21, (512, 512))

img22 = cv.GaussianBlur(img1, (5, 5), 0)

img23 = cv.GaussianBlur(img8, (15, 15), 0)

img24 = cv.addWeighted(img22, 0.9, img23, 0.8, 0)

cv.imshow("img1", img1)
# cv.imshow("img4", img4)
# cv.imshow("img5", img5)
# cv.imshow("img5_hsv", img5_hsv)
# cv.imshow("img7", img7)
cv.imshow("img8", img8)
cv.imshow("img9", img9)
cv.imshow("img10", img10)
cv.imshow("img11", img11)
cv.imshow("img13", img13)
cv.imshow("img14", img14)
# cv.imshow("img15", img15)
# cv.imshow("img16", img16)
cv.imshow("img17", img17)
cv.imshow("img18", img18)
# cv.imshow("img19", img19)
# cv.imshow("img20", img20)
cv.imshow("img21", img21)
cv.imshow("img22", img22)
cv.imshow("img23", img23)
cv.imshow("img24", img24)
cv.waitKey(0)
cv.destroyAllWindows()

cv.imwrite("1_2.jpg", img1)
cv.imwrite("2_2.jpg", img21)

