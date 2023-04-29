import cv2

img = cv2.imread('ollie_shiloute.png', cv2.IMREAD_UNCHANGED)

resized = cv2.resize(img, (28,28), interpolation = cv2.INTER_AREA)

cv2.imwrite('ollie_resized.png', resized)