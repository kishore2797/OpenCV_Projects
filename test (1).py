import cv2 as cv
import numpy as np
import random
import matplotlib.pyplot as plt

original_img = cv.imread("lena.jpg")

# print(original_img)
print(np.shape(original_img))

gray_img = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)

# print(gray_img)

print(np.shape(gray_img))



cv.imshow("original_img", original_img)
cv.imshow("gray_img", gray_img)
cv.waitKey(0)
cv.destroyAllWindows()

# convert BGR to RGB format
# opencv reads image in BGR format
# matplotlib shows image in RBG fromat, so convert BGR to RGB for plotting purpose

matplot_img = cv.cvtColor(original_img, cv.COLOR_BGR2RGB)

img = np.copy(original_img)

# cv.imshow("Image Windows", img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# print(img.shape)  # returns a tuple of no.of rows, columns and channels ::: (512, 512) -> 0, (512, 512, 3) -> 1,-1
# print(img.size)  # returns total no.of pixels is accessed
# print(img.dtype)  # returns  image data type is obtined ::: GRAY and COLOR -> uint8

# 3 channels -> red, green, blue

# read the channels and split
# in gray scale error has been thrown -> ValueError: not enough values to unpack (expected 3, got 1)
# because in gray scale their is only one channel. 0 or 1
b, g, r = cv.split(img)
# print(f"b {b},\n g {g},\n r {r}\n")  # split the three matrices or pixels matrices

# print(b.shape)
# print(b.dtype)

zeros = np.zeros((512, 512), dtype="uint8")

# print(zeros)

# fill the blue matrix with zeros i.e., change the color of blue to 0. like -> #ffff00 (rgb format)

b1 = np.zeros((512, 512), dtype="uint8")

# now combine or merge the matrices into one

img1 = cv.merge((b1, g, r))  # merge the color channels as tuples

# now see the results of both img and img1

# cv.imshow("img", img)
# cv.imshow("img1", img1)
# cv.waitKey(0)
# cv.destroyAllWindows()

# now fill the green matrix with zeros

g1 = np.zeros((512, 512), dtype="uint8")

img2 = cv.merge((b, g1, r))

# now fill the red matrix with zeros

r1 = np.zeros((512, 512), dtype="uint8")

img3 = cv.merge((b, g, r1))

# now try combinations of zeros matrices

img4 = cv.merge((b, g1, r1))
img5 = cv.merge((b1, g, r1))
img6 = cv.merge((b1, g1, r))

# now try all zeros matrix

img7 = cv.merge((b1, g1, r1))


# cv.imshow("img", img)
# cv.imshow("img1", img1)
# cv.imshow("img2", img2)
# cv.imshow("img3", img3)
# cv.imshow("img4", img4)
# cv.imshow("img5", img5)
# cv.imshow("img6", img6)
# cv.imshow("img7", img7)
# cv.waitKey(0)
# cv.destroyAllWindows()

# now flip the img

def horizontalFlip(olg_img, flag):
    if flag:
        return cv.flip(olg_img, 1)
    else:
        return olg_img


horiImg = horizontalFlip(img, True)


def verticalFlip(olg_img, flag):
    if flag:
        return cv.flip(olg_img, -1)
    else:
        return olg_img


verImg = verticalFlip(img, True)

# flipCode greater than 0 -> horizontalFlip
# flipCode less than or equal to 0 -> verticalFlip

# flip means transpose the matrix or inverse the matrix

# cv.imshow("img", img)
# cv.imshow("horiImg", horiImg)
# cv.imshow("verImg", verImg)
# cv.waitKey(0)
# cv.destroyAllWindows()

# now try reflect or mirror of img

ratio = 0.8

# img [height : width : channel] -> img[512: 512: 3] for color channel

height, width = img.shape[:2]

# print(height, width)

to_shift = int(ratio * width)

# img [height, width - to_shift, :] or [:, width-to_shift, :]

reflectImg = cv.copyMakeBorder(img, 0, 0, width - to_shift, height, cv.BORDER_REFLECT)

# print(reflectImg.shape)

# reshape the img

newReflectImg = np.reshape(img, [512, 512, 3])

# print(newReflectImg.shape)

# now try to create 512x512 reflective img

rHeight, rWidth = img.shape[:2]

r_to_shift_ratio = int(rWidth * ratio)

rImg = img[:, :rWidth - r_to_shift_ratio, :]

rImg = cv.copyMakeBorder(rImg, 0, 0, r_to_shift_ratio, 0, cv.BORDER_REFLECT)

# cv.imshow("img", img)
# cv.imshow("reflectImg", reflectImg)
# cv.imshow("newReflectImg", newReflectImg)
# cv.imshow("rImg", rImg)
# cv.waitKey(0)
# cv.destroyAllWindows()

# above all are wrong reflect image methods

ratio1 = 0.5

new_ratio1 = random.uniform(-ratio1, ratio1)

# print(new_ratio1)

to_shift_1 = int(rWidth * ratio1)

reflectImg1 = img[:, :rWidth - to_shift_1, :]

newReflectImg1 = cv.copyMakeBorder(reflectImg1, 0, 0, to_shift_1, 0, cv.BORDER_REFLECT)

# cv.imshow("img", img)
# cv.imshow("newReflectImg1", newReflectImg1)
# cv.waitKey(0)
# cv.destroyAllWindows()

# cartoon img

img8 = np.copy(img)

cv.stylization(img8, img8, 100, 0.1)

# cv.imshow("img", img)
# cv.imshow("img8", img8)
# cv.waitKey(0)
# cv.destroyAllWindows()

# pencil sketch img

# gray scale img
gray_scale_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# invert img
invert_img = 255 - gray_scale_img

# blue the invert img
# create the kernel size -> GaussianBlue(src, kernel_size, sigma)

kernel_size = (7, 7)

# (7, 7) -> (width, height) ::: width and height of the kernel which should be positive and odd

# we also should specify the standard deviation in the X and Y directions, sigmaX and sigmaY respectively.
# If only sigmaX is specified, sigmaY is taken as equal to sigmaX.
# If both are given as zeros, they are calculated from the kernel size

# Gaussian filtering is highly effective in removing Gaussian noise from the image.

blur_img = cv.GaussianBlur(invert_img, kernel_size, 0)

# https://en.wikipedia.org/wiki/Blend_modes
# The Colour Dodge blend mode divides the bottom layer by the inverted top layer.
# This lightens the bottom layer depending on the value of the top layer.
# We have the blurred image, which highlights the boldest edges.

# cv.imshow("img", img)
# cv.imshow("gray_scale_img", gray_scale_img)
# cv.imshow("invert_img", invert_img)
# cv.imshow("blur_img", blur_img)
# cv.waitKey(0)
# cv.destroyAllWindows()

img12 = blur_img * 255
img13 = 255 - gray_scale_img
img14 = img12 / invert_img
img15 = np.copy(img14)
img15[img15 > 255] = 255

pencil_sketch_img = np.copy(img15)

pencil_sketch_img[gray_scale_img == 255] = 255

# cv.imshow("img", img)
# cv.imshow("blur_img", blur_img)
# cv.imshow("img12", img12)
# cv.imshow("gray_scale_img", gray_scale_img)
# cv.imshow("img13", img13)
# cv.imshow("img14", img14)
# cv.imshow("img15", img15)
# cv.imshow("pencil_sketch_img", pencil_sketch_img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# cv.imwrite("pencil_sketch.png", pencil_sketch_img)
# cv.imwrite("pencil_sketch.png", gray_scale_img)

# print(gray_scale_img)
#
# print(blur_img)
#
# print(pencil_sketch_img)

# pencil_sketch_img[pencil_sketch_img > 1] = 255
# print(pencil_sketch_img)

img16 = np.copy(pencil_sketch_img)

img17 = ((img16 - img16.min()) * (1 / (img16.max() - img16.min()) * 255)).astype("uint8")

# print(img17)

img18 = cv.normalize(img16, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_32F).astype("uint8")

# cv.imshow("pencil_sketch_img", pencil_sketch_img)
# cv.imshow("img17", img17)
# cv.imshow("img18", img18)
# cv.waitKey(0)
# cv.destroyAllWindows()

# print(img16)
# print(img16.shape)
# print(img16.size)

# print(img16[0])

# print(img16.astype("uint8"))

# print(img16 * 100)

img19 = (img16 * 25).astype("uint8")

# print(img19)

img20 = 255 - img8

img21 = cv.cvtColor(gray_scale_img, cv.COLOR_GRAY2BGR)

# print(img21.shape)

b, g, r = cv.split(img21)

img22 = cv.merge((b1, g, r))
img23 = cv.merge((b, g1, r))
img24 = cv.merge((b, g, r1))
img25 = cv.merge((b1, g1, r))
img26 = cv.merge((b1, g, r1))
img27 = cv.merge((b, g1, r1))

# cv.imshow("img16", img16)
# cv.imshow("img18", img18)
# cv.imshow("img19", img19)
# cv.imshow("img21", img21)
# cv.imshow("img22", img22)
# cv.imshow("img23", img23)
# cv.imshow("img24", img24)
# cv.imshow("img25", img25)
# cv.imshow("img26", img26)
# cv.imshow("img27", img27)
# cv.waitKey(0)
# cv.destroyAllWindows()

from scipy import ndimage as nd_image

blur_img_1 = nd_image.gaussian_filter(invert_img, sigma=2)

# more blurring on increasing sigma

# print(blur_img)
# print(blur_img_1)

# print(invert_img)

pencil_sketch_img_1 = (blur_img_1 * 25) / invert_img
# print(blur_img_1)
# print((blur_img_1 * 4)[0])
# print(pencil_sketch_img_1)
pencil_sketch_img_1[pencil_sketch_img_1 > 255] = 255
pencil_sketch_img_1[gray_scale_img == 255] = 255

# print(pencil_sketch_img)
# print(pencil_sketch_img_1)

img28 = (pencil_sketch_img_1 * 100).astype("uint8")
img29 = cv.cvtColor(img28, cv.COLOR_GRAY2BGR)

b, g, r = cv.split(img29)

img30 = cv.merge((b1, g, r))

# cv.imshow("pencil_sketch_img", pencil_sketch_img)
# cv.imshow("pencil_sketch_img_1", pencil_sketch_img_1)
# cv.imshow("img29", img29)
# cv.imshow("img30", img30)
# cv.waitKey(0)
# cv.destroyAllWindows()

# images = [img18, pencil_sketch_img, pencil_sketch_img_1]

# images = [invert_img, blur_img, blur_img_1]
#
# for i in range(len(images)):
#     plt.subplot(2, 2, i + 1)
#     plt.imshow(images[i], cmap='gray')
# plt.show()

height, width = img.shape[:2]

img31 = cv.cvtColor(img, cv.CV_8UC1)

img31_gray = cv.cvtColor(img31, cv.COLOR_BGR2GRAY)

img31_invert = 255 - img31_gray

img31_blur = cv.GaussianBlur(img31_invert, (21, 21), 0, 0)

# src1 = img31_gray
# src2 = img31_blur

# print(img31_gray.shape)
# print(img31_blur.shape)

img32 = cv.divide(img31_gray, img31_blur, scale=256)

# print(img32)

img33 = 255 - cv.divide(255 - img31_gray, 255 - img31_blur, scale=256)

img34 = cv.multiply(img32, img31_gray, scale=1 / 256)

# cv.imshow("img", img)
# cv.imshow("img31", img31)
# cv.imshow("img31_gray", img31_gray)
# cv.imshow("img31_invert", img31_invert)
# cv.imshow("img31_blur", img31_blur)
# cv.imshow("img32", img32)
# cv.imshow("img33", img33)
# cv.imshow("img34", img34)
# cv.waitKey(0)
# cv.destroyAllWindows()

# cv.imwrite("lena_gray.png", img32)

img35 = np.zeros((240, 320, 3), np.uint8)
img35 = cv.rectangle(img35, (200, 0), (300, 200), (0, 255, 255), -1)

# rectangle(src, pt1, pt2, color, thickness)

img36 = cv.imread("LinuxLogo.jpg")

# print(img36.shape)

bitAnd = cv.bitwise_and(img36, img35)
bitOr = cv.bitwise_or(img36, img35)
bitXor = cv.bitwise_xor(img36, img35)
bitNot = cv.bitwise_not(img35)


# print(img36)
# print(bitXor)

# | AND Table |
# | B | A | Result |
# | 0 | 0 |   0    |
# | 0 | 1 |   0    |
# | 1 | 0 |   0    |
# | 1 | 1 |   1    |

# | OR Table |
# | B | A | Result |
# | 0 | 0 |   0    |
# | 0 | 1 |   1    |
# | 1 | 0 |   1    |
# | 1 | 1 |   1    |

# | XOR Table |
# | B | A | Result |
# | 0 | 0 |   0    |
# | 0 | 1 |   1    |
# | 1 | 0 |   1    |
# | 1 | 1 |   0    |

# | NOT Table |
# | B | Result |
# | 0 | 1      |
# | 0 | 0      |

# cv.imshow("img35", img35)
# cv.imshow("img36", img36)
# cv.imshow("bitAnd", bitAnd)
# cv.imshow("bitOr", bitOr)
# cv.imshow("bitXor", bitXor)
# cv.imshow("bitNot", bitNot)
# cv.waitKey(0)
# cv.destroyAllWindows()

# HSV -> Hue, Saturation, Value ::: Ex:(160, 50%, 70%)

# frame = cv.imread("opencv-logo.png")
#
# hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
#
# print(frame)
# print(hsv)
#
# lower_bound = np.array([110, 50, 50], np.uint8)
# upper_bound = np.array([130, 255, 255], np.uint8)
#
# mask = cv.inRange(hsv, lower_bound, upper_bound)
#
# res = cv.bitwise_and(frame, frame, mask=mask)
#
# cv.imshow("frame", frame)
# cv.imshow("hsv", hsv)
# cv.imshow("mask", mask)
# cv.imshow("res", res)
# cv.waitKey(0)
# cv.destroyAllWindows()


def nothing(x):
    pass


# cv.namedWindow("Tracking")
# cv.createTrackbar("LH", "Tracking", 0, 255, nothing)  # LH -> Lower Hue
# cv.createTrackbar("LS", "Tracking", 0, 255, nothing)  # LS -> Lower Saturation
# cv.createTrackbar("LV", "Tracking", 0, 255, nothing)  # LV -> Lower Value
# cv.createTrackbar("UH", "Tracking", 255, 255, nothing)  # UH -> Upper Hue
# cv.createTrackbar("US", "Tracking", 255, 255, nothing)  # US -> Upper Saturation
# cv.createTrackbar("UV", "Tracking", 255, 255, nothing)  # UV -> Upper Value

# while True:
#     frame = cv.imread("opencv-logo.png")
#
#     hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
#
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
#     lower_bound_2 = np.array([120, 50, 255], np.uint8)
#     upper_bound_2 = np.array([120, 255, 255], np.uint8)
#
#     mask = cv.inRange(hsv, lower_bound, upper_bound)
#     mask_2 = cv.inRange(hsv, lower_bound_2, upper_bound_2)
#
#     res = cv.bitwise_and(frame, frame, mask=mask)
#
#     res2 = cv.bitwise_and(frame, frame, mask=mask_2)
#
#     cv.imshow("frame", frame)
#     cv.imshow("hsv", hsv)
#     cv.imshow("mask", mask)
#     cv.imshow("res", res)
#     cv.imshow("mask_2", mask_2)
#     cv.imshow("res2", res2)
#
#     key = cv.waitKey(1)
#     if key == 27:
#         break
#
# cv.destroyAllWindows()

# while True:
#     frame = cv.imread("HappyFish.jpg")
#
#     hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
#
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
#     lower_bound_2 = np.array([120, 50, 255], np.uint8)
#     upper_bound_2 = np.array([120, 255, 255], np.uint8)
#
#     mask = cv.inRange(hsv, lower_bound, upper_bound)
#     mask_2 = cv.inRange(hsv, lower_bound_2, upper_bound_2)
#
#     res = cv.bitwise_and(frame, frame, mask=mask)
#
#     res2 = cv.bitwise_and(frame, frame, mask=mask_2)
#
#     cv.imshow("frame", frame)
#     cv.imshow("hsv", hsv)
#     cv.imshow("mask", mask)
#     cv.imshow("res", res)
#     cv.imshow("mask_2", mask_2)
#     cv.imshow("res2", res2)
#
#     key = cv.waitKey(1)
#     if key == 27:
#         break
#
# cv.destroyAllWindows()

# threshold technique

img37 = cv.imread("opencv-logo.png", 0)

# _, th1 = cv.threshold(img37, 127, 255, cv.THRESH_BINARY)
# _, th2 = cv.threshold(img37, 157, 255, cv.THRESH_BINARY_INV)
# _, th3 = cv.threshold(img37, 157, 255, cv.THRESH_TRUNC)
# _, th4 = cv.threshold(img37, 0, 255, cv.THRESH_TOZERO)
# _, th5 = cv.threshold(img37, 180, 255, cv.THRESH_TOZERO_INV)

# cv.imshow("img37", img37)
# cv.imshow("th1", th1)
# cv.imshow("th2", th2)
# cv.imshow("th3", th3)
# cv.imshow("th4", th4)
# cv.imshow("th5", th5)
# cv.waitKey(0)
# cv.destroyAllWindows()


img38 = cv.imread("lena.jpg", 0)

# cv.namedWindow("Tracking")
# cv.createTrackbar("Thresh", "Tracking", 0, 255, nothing)
#
# while True:
#     thresh = cv.getTrackbarPos("Thresh", "Tracking")
#
#     _, th1 = cv.threshold(img38, thresh, 255, cv.THRESH_BINARY)
#     _, th2 = cv.threshold(img38, thresh, 255, cv.THRESH_BINARY_INV)
#     _, th3 = cv.threshold(img38, thresh, 255, cv.THRESH_TRUNC)
#     _, th4 = cv.threshold(img38, thresh, 255, cv.THRESH_TOZERO)
#     _, th5 = cv.threshold(img38, thresh, 255, cv.THRESH_TOZERO_INV)
#
#     cv.imshow("img38", img38)
#     cv.imshow("th1", th1)
#     cv.imshow("th2", th2)
#     cv.imshow("th3", th3)
#     cv.imshow("th4", th4)
#     cv.imshow("th5", th5)
#
#     key = cv.waitKey(1)
#
#     if key == 27:
#         break
#
# cv.destroyAllWindows()

