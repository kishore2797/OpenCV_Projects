import cv2
import numpy as np

# load image
img = cv2.imread("1_2.jpg")

img1 = img.copy()

# apply morphology open to smooth the outline
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
morph = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# brighten dark regions
result = cv2.normalize(morph,None,20,255,cv2.NORM_MINMAX)

# write result to disk
cv2.imwrite("windmill_oilpaint.jpg", result)



cv2.imshow("IMAGE", img)
cv2.imshow("OPEN", morph)
cv2.imshow("RESULT", result)
cv2.imshow("img", cv2.morphologyEx(img1, cv2.MORPH_TOPHAT, (7, 7)))
cv2.waitKey(0)
cv2.destroyAllWindows()