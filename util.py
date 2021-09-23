import cv2 as cv


def createAndReturnPencilSketch(color_img, ksize, blur):
    gray_img = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)
    invert_gray_img = 255 - gray_img
    blur_img = cv.GaussianBlur(invert_gray_img, ksize, blur)

    pencil_sketch = cv.divide(gray_img, blur_img, scale=256)

    return pencil_sketch


def createAndReturnPencilSketchLight(color_img, ksize, blur):
    gray_img = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)
    invert_gray_img = 255 - gray_img
    blur_img = cv.GaussianBlur(invert_gray_img, ksize, blur)

    pencil_sketch = cv.divide(gray_img, blur_img, scale=256)

    pencil_sketch_light = cv.multiply(pencil_sketch, gray_img, scale= 1/256)

    return pencil_sketch_light


def createAndReturnPencilSketchDark(color_img, ksize, blur):
    gray_img = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)
    invert_gray_img = 255 - gray_img
    blur_img = cv.GaussianBlur(invert_gray_img, ksize, blur)

    pencil_sketch = 255 - cv.divide(255 - gray_img, 255 - blur_img, scale=256)

    return pencil_sketch


def createAndSavePencilSketch(color_img, ksize, blur, filename="pencil_sketch"):
    gray_img = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)
    invert_gray_img = 255 - gray_img
    blur_img = cv.GaussianBlur(invert_gray_img, ksize, blur)

    pencil_sketch = cv.divide(gray_img, blur_img, scale=256)

    cv.imwrite(filename, pencil_sketch)

# testing

# img = cv.imread("2.jpeg")
#
# img1 = createAndReturnPencilSketch(img, (23, 23), 0)
#
# cv.imshow("img", img)
# cv.imshow("img1", img1)
# cv.waitKey(0)
# cv.destroyAllWindows()