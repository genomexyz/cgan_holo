import cv2
import numpy as np

#setting
dir_save = 'ollie'
mean = 0
var = 30
sigma = var ** 0.5
total_data = 640

img = cv2.imread('ollie_resized.png', cv2.IMREAD_UNCHANGED)
img = img[:,:,:3]

#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

for i in range(total_data):
    gaussian = np.random.normal(mean, sigma, (img.shape[0],img.shape[1],img.shape[2]))
    img_noise = img + gaussian
    cv2.imwrite('%s/data%s.png'%(dir_save, i), img_noise)