import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# try to create realistic cartoon img
# test the original img pixles and cartoon images pixels
# for how they difference and try to create another using the pixels

img = cv.imread("1.jpg")
img1 = cv.imread("2.jpeg")

print(img.shape)

# cv.imshow("img", img)
# cv.waitKey(0)
# cv.destroyAllWindows()

img2 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img3 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)

# print(img1)

# print(img2.shape)
# print(img3.shape)

# print(img2[0].shape)
# print(img2[1].shape)

# print(img2[0][23])

# print(img2[0][0])
# print(len(img2))
# print(len(img2[0]))
# print(len(img2[0][0]))

height, width = img2.shape[:2]

# print(img2[345][277])
# print(img3[345][277])

img4 = np.empty((height, width, 3), dtype=np.uint8)

# print(img4)

# img4[0][0] = [1, 0, 0]
#
# print(img4[0])

for i in range(len(img2)):
    for j in range(len(img2[i])):
        img4[i][j] = img2[i][j]

img5 = np.empty((height, width, 3), dtype=np.uint8)

for i in range(len(img2)):
    for j in range(len(img2[i])):
        if i < 500:
            img5[i][j] = img2[i][j]
        else:
            img5[i][j] = img3[i][j]

# img6 = np.empty((height, width, 3), dtype=np.uint8)
#
# for i in range(len(img2)):
#     for j in range(len(img2[i])):
#         for k in range(len(img2[i][j])):
#             img6[i][j][k] = img3[i][j][k] - img2[i][j][k]

hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
hsv_img1 = cv.cvtColor(img1, cv.COLOR_BGR2HSV)

# titles = ["img2", "img3", "img4", "img5", "hsv_img", "hsv_img1"]
# images = [img2, img3, img4, img5, hsv_img, hsv_img1]
#
# for i in range(len(images)):
#     plt.subplot(2, 4, i+1)
#     plt.imshow(images[i], "gray")
#     plt.title(titles[i])
#
# plt.show()

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
#     mask = cv.inRange(hsv_img1, lower_bound, upper_bound)
#
#     res = cv.bitwise_and(img1, img1, mask=mask)
#
#     cv.imshow("mask", mask)
#     cv.imshow("res", res)
#
#     key = cv.waitKey(1)
#     if key == 27:
#         break
#
# cv.destroyAllWindows()

# r = 250 - 255 , g = 170 - 120, b = 110 - 180

img6 = np.empty((height, width, 3), dtype=np.uint16)

# for i in range(len(img3)):
#     for j in range(len(img3[i])):
#         for k in range(len(img3[i][j])):
#             if k == 0:
#                 if 190 < img3[i][j][k] < 256:
#                     img6[i][j][k] = img2[i][j][k]
#                 else:
#                     img6[i][j][k] = img3[i][j][k]
#             if k == 1:
#                 if 110 < img3[i][j][k] < 210:
#                     img6[i][j][k] = img2[i][j][k]
#                 else:
#                     img6[i][j][k] = img3[i][j][k]
#             if k == 2:
#                 if 70 < img3[i][j][k] < 190:
#                     img6[i][j][k] = img2[i][j][k]
#                 else:
#                     img6[i][j][k] = img3[i][j][k]

# titles = ["img3", "img2", "img6"]
# images = [img3, img2, img6]
#
# for i in range(len(images)):
#     plt.subplot(2, 2, i + 1)
#     plt.imshow(images[i], "gray")
#     plt.title(titles[i])
#
# plt.show()


