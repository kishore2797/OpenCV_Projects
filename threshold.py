import cv2 as cv
import numpy as np

img = cv.imread("1_2.jpg")

# adaptive threshold

img1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# img1 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
# img2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, 2)
img3 = cv.adaptiveThreshold(img1, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 7, 6)
# img4 = cv.adaptiveThreshold(img3, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 23, 8)

img5 = cv.imread("neon_1.jpg")

img5 = cv.resize(img5, (512, 512))

img6 = cv.cvtColor(img3, cv.COLOR_GRAY2RGB)

img7 = cv.bitwise_or(img6, img5)

img8 = cv.blur(img7, (3, 3))

img9 = img * img8

# cv.imshow("img", img)
# cv.imshow("img1", img1)
# cv.imshow("img2", img2)
# cv.imshow("img3", img3)
# cv.imshow("img4", img4)
# cv.imshow("img5", img5)
# cv.imshow("img6", img6)
cv.imshow("img7", img7)
cv.imshow("img8", img8)
cv.imshow("img9", img9)
cv.waitKey(0)
cv.destroyAllWindows()

# img8 = cv.resize(img, (250, 250), interpolation=cv.INTER_CUBIC)
#
# cv.imshow("img8", img8)
# cv.waitKey(0)
# cv.destroyAllWindows()

rows, cols = img7.shape[:2]

m = np.float32([[1, 0, 5], [0, 1, 4]])

img8 = cv.warpAffine(img7, m, (cols, rows))

img9 = cv.imread("2.jpeg")
img9 = cv.resize(img9, (512, 512))

img10 = cv.GaussianBlur(img9, (5, 5), 0)

img11 = cv.bitwise_and(img9, img10)

img12 = cv.add(img10, img9)

img13 = cv.medianBlur(img12, 9)

img14 = cv.medianBlur(img9, 9)

img15 = cv.cvtColor(img3, cv.COLOR_GRAY2RGB)

img16 = cv.bitwise_or(img14, img15)

img17 = cv.bitwise_and(img14, img16)

# img4
#
# img18 = cv.cvtColor(img4, cv.COLOR_GRAY2RGB)
#
# img19 = cv.bitwise_or(img14, img18)
#
# img20 = cv.bitwise_and(img14, img19)
#
# img21 = cv.addWeighted(img16, 0.2, img16, 0.8, 0)
#
# img22 = cv.bitwise_and(img19, img21)
#
# img23 = cv.addWeighted(img19, 0.4, img19, 0.8, 0)
#
# img24 = cv.bitwise_and(img23, img21)
#
# img25 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 17, 10)
#
# img25 = cv.cvtColor(img25, cv.COLOR_GRAY2RGB)
#
# img26 = cv.bitwise_or(img25, img10)
#
# img27 = cv.bitwise_and(img26, img24)
#
# img28 = cv.add(img9, img27)


def nothing(x):
    pass


# cv.namedWindow("Tracking")
# cv.createTrackbar("alpha1", "Tracking", 0, 10, nothing)
# cv.createTrackbar("alpha2", "Tracking", 0, 10, nothing)
#
# while True:
#     alpha1 = cv.getTrackbarPos("alpha1", "Tracking") / 10
#     alpha2 = cv.getTrackbarPos("alpha2", "Tracking") / 10
#
#     img29 = cv.addWeighted(img9, alpha1, img27, alpha2, 0)
#
#     cv.imshow("img7", img7)
#     cv.imshow("img15", img15)
#     cv.imshow("img16", img16)
#     cv.imshow("img19", img19)
#     cv.imshow("img21", img21)
#     cv.imshow("img22", img22)
#     cv.imshow("img23", img23)
#     cv.imshow("img25", img25)
#     cv.imshow("img26", img26)
#     cv.imshow("img27", img27)
#     cv.imshow("img29", img29)
#
#     key = cv.waitKey(1)
#     if key == 27:
#         break
#
# cv.destroyAllWindows()

# cv.imshow("img3", img3)
# cv.imshow("img7", img7)
# cv.imshow("img8", img8)
# cv.imshow("img9", img9)
# cv.imshow("img10", img10)
# cv.imshow("img11", img11)
# cv.imshow("img12", img12)
# cv.imshow("img13", img13)
# cv.imshow("img14", img14)
# cv.imshow("img15", img15)
# cv.imshow("img16", img16)
# cv.imshow("img17", img17)
# cv.imshow("img19", img19)
# cv.imshow("img20", img20)
# cv.imshow("img21", img21)
# cv.imshow("img22", img22)
# cv.imshow("img23", img23)
# cv.imshow("img24", img24)
# cv.imshow("img25", img25)
# cv.imshow("img26", img26)
# cv.imshow("img27", img27)
# cv.imshow("img28", img28)
# cv.imshow("img29", img29)
# cv.waitKey(0)
# cv.destroyAllWindows()
