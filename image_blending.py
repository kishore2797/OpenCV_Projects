import cv2 as cv
import numpy as np
import util

# arithmetic operations on images

# cv.add(), cv.addWeighted(), etc.,

# add two images by open-cv function, cv.add() or
# simply by numpy operation res = img + img1
# Note : Both images should be of same depth and type or second images can just be a scalar value

x = np.uint8([250])
y = np.uint8([10])

print(cv.add(x, y)) # o/p : [[255]]

print(x + y) # o/p : [4]

# there is a difference b/w open-cv addition and numpy addition.
# open-cv addition is a saturated operation
# while numpy addition is a modulo operation.

img = cv.imread("lena.jpg")
img1 = cv.imread("lena_gray.png")

print(img.shape, img1.shape)

img2_hsv = cv.cvtColor(img1, cv.COLOR_BGR2HSV)

# open-cv add function
img3 = cv.add(img, img2_hsv)
img4 = cv.add(img, img1)
img5 = cv.add(img1, img2_hsv)
img6 = cv.add(img3, img1)
img7 = cv.add(img, img6)
img8 = cv.add(img6, img)

# cv.add(a, b) and cv.add(b, a) are same

# numpy normal addition
img9 = img + img2_hsv

img10 = cv.bitwise_or(img, img6)

# cv.imshow("img", img)
# cv.imshow("img1", img1)
# cv.imshow("img2_hsv", img2_hsv)
# cv.imshow("img3", img3)
# cv.imshow("img4", img4)
# cv.imshow("img5", img5)
# cv.imshow("img6", img6)
# cv.imshow("img7", img7)
# cv.imshow("img8", img8)
# cv.imshow("img10", img10)
# cv.waitKey(0)
# cv.destroyAllWindows()


# def nothing(x):
#     pass
#
#
# cv.namedWindow("Tracking")
# cv.createTrackbar("LH", "Tracking", 0, 255, nothing)  # LH -> Lower Hue
# cv.createTrackbar("LS", "Tracking", 0, 255, nothing)  # LS -> Lower Saturation
# cv.createTrackbar("LV", "Tracking", 0, 255, nothing)  # LV -> Lower Value
# cv.createTrackbar("UH", "Tracking", 255, 255, nothing)  # UH -> Upper Hue
# cv.createTrackbar("US", "Tracking", 255, 255, nothing)  # US -> Upper Saturation
# cv.createTrackbar("UV", "Tracking", 255, 255, nothing)  # UV -> Upper Value
#
# while True:
#     l_h = cv.getTrackbarPos("LH", "Tracking")
#     l_s = cv.getTrackbarPos("LS", "Tracking")
#     l_v = cv.getTrackbarPos("LV", "Tracking")
#
#     u_h = cv.getTrackbarPos("UH", "Tracking")
#     u_s = cv.getTrackbarPos("US", "Tracking")
#     u_v = cv.getTrackbarPos("UV", "Tracking")
#
#     lower_bound = np.array([l_h, l_s, l_v], np.uint8)
#     upper_bound = np.array([u_h, u_s, u_v], np.uint8)
#
#     mask = cv.inRange(img6, lower_bound, upper_bound)
#
#     res = cv.bitwise_and(img, img, mask=mask)
#
#     cv.imshow("mask", mask)
#     cv.imshow("res", res)
#
#     key = cv.waitKey(1)
#     if key == 27:
#         break
#
# cv.destroyAllWindows()

# open-cv function will provide a better result.
# So always better stick to open-cv functions


# Image Blending

# formula g(x) = (1- a) f0(x) + af1(x) ::: a -> alpha
# this is also image addition, but different weights are given to images
# so that it given a feeling of blending or transparency.
# Images are added as per the equation above.
# by varying a from 0 --> 1, we can perform a cool transition b/w one image to another

# Now took two images to blend them together.
# First image is given a weight of 0.7 and
# second image is given 0.3
# cv.addWeighted() -- applies following equ on the img.
# dst = a x img + b x img1 + r  ::: a -> alpha, b -> beta, r -> gamma
# here r is taken as zero.

dst = cv.addWeighted(img, 1.0, img6, 0.4, 0)
dst1 = cv.addWeighted(img, 1.0, img1, 0.4, 0)

# cv.imshow("img", img)
# cv.imshow("img1", img1)
# cv.imshow("img6", img6)
# cv.imshow("dst", dst)
# cv.imshow("dst1", dst1)
# cv.waitKey(0)
# cv.destroyAllWindows()