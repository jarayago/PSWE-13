import cv2 as cv
img = cv.imread("IMG_PATH")

cv.imshow("Display window", img)
k = cv.waitKey(0) # Wait for a keystroke in the window