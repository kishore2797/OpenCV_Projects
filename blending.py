import cv2 as cv
import numpy as np

# img = cv.imread("1_2.jpg")
# img1 = cv.imread("2_2.jpg")
#
# img2 = cv.pyrDown(img)
# img3 = cv.pyrUp(img2)
#
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# # blend two images
# img_img1 = np.hstack((img[:, :256], img1[:, 256:]))
#
# cv.imshow("img", img)
# cv.imshow("img1", img1)
# cv.imshow("img_img1", img_img1)
# cv.imshow("img_subtract", cv.subtract(img, img3))
# cv.imshow("img2", img2)
# cv.imshow("img3", img3)
# cv.waitKey(0)
# cv.destroyAllWindows()


# gaussian pyramids using pyrDown and pyrUp
# laplacian pyramids
# there is no methods to create laplacian pyramids
# A level in laplacian pyramid is formed by the difference between that level in Gaussian pyramid
# and expanded version of its upper level in Gaussian pyramid

# 5 steps to blend the images properly

# step-1-> load the two images.
# step-2-> find the gaussian pyramids for both images ( no.of levels is 6 ).
# step-3-> from gaussian pyramids, find their laplacian pyramids.
# step-4-> now join the left half of first img and right half of second img is each levels of laplacian pyramids.
# step-5-> finally from this joint image pyramids, reconstruct the original image.


# step-1: Load the images.

img = cv.imread("1_2.jpg")
img1 = cv.imread("2_2.jpg")

# step-2-> find the gaussian pyramids for both images ( no.of levels is 6 ).

img_copy = img.copy()
img1_copy = img1.copy()

gp_img = [img_copy]
gp_img1 = [img1_copy]

for i in range(6):
    img_copy = cv.pyrDown(img_copy)
    img1_copy = cv.pyrDown(img1_copy)
    gp_img.append(img_copy)
    gp_img1.append(img1_copy)
    # cv.imshow(str(i), img_copy)
    # cv.imshow(str(i) + "1", img1_copy)

# step-3-> from gaussian pyramids, find their laplacian pyramids.

lp_img = [gp_img[6]]
lp_img1 = [gp_img1[6]]

for i in range(6, 0, -1):
    gaussian_extended = cv.pyrUp(gp_img[i])
    gaussian_extended_1 = cv.pyrUp(gp_img1[i])
    laplacian = cv.subtract(gp_img[i-1], gaussian_extended)
    laplacian1 = cv.subtract(gp_img1[i - 1], gaussian_extended_1)
    lp_img.append(laplacian)
    lp_img1.append(laplacian1)
    # cv.imshow(str(i), laplacian)
    # cv.imshow(str(i)+"1", laplacian1)

# step-4-> now join the left half of first img and right half of second img is each levels of laplacian pyramids.

lp_img_img1_blend = list()

for i in range(len(lp_img)):
    blend = np.hstack((lp_img[i][:, :256], lp_img1[i][:, 256:]))
    lp_img_img1_blend.append(blend)
    # cv.imshow(f"blend {i}", blend)


# cv.imshow("blend", lp_img_img1_blend[6])

img_img1_recontruct = lp_img_img1_blend[0]

for i in range(1, 7):
    img_img1_recontruct = cv.pyrUp(img_img1_recontruct)
    cv.imshow("img_img1_recontruct_" + str(i), img_img1_recontruct)
    img_img1_recontruct = cv.add(lp_img_img1_blend[i], img_img1_recontruct)

cv.imshow("img_img1_recontruct", img_img1_recontruct)
cv.waitKey(0)
cv.destroyAllWindows()