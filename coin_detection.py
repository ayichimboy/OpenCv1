#import libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils

#call image
coin = cv2.imread("coin.jpg")
img_coin = cv2.medianBlur(coin, 5)
plt.title("Coins")
plt.imshow(img_coin)
#plt.show()

#convert to grayscale
img_coin_gray = cv2.cvtColor(img_coin, cv2.COLOR_BGR2GRAY)
plt.title("gray")
plt.imshow(img_coin_gray)
#plt.show()

#gamma correction
gray_cor_coin = np.array(255 * (img_coin_gray/255)** 1.2, dtype='uint8')
plt.title("Gamma")
plt.imshow(gray_cor_coin)
#plt.show()

#histogram
gray_equ=cv2.equalizeHist(img_coin_gray)
plt.title("Hist")
plt.imshow(gray_equ)
#plt.show()

#local adaptive scale
#global thresholding
ret, th1 = cv2.threshold(img_coin_gray, 127, 255, cv2.THRESH_BINARY)
#otsu thresholding
ret2, th2 = cv2.threshold(img_coin_gray,0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#adaptive thresholding
hold = cv2.adaptiveThreshold(img_coin_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 255, 19)
#
hold_1 = cv2.adaptiveThreshold(img_coin_gray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 19)

hold=cv2.bitwise_not(hold)
#blur = cv2.GaussianBlur(coin, (5,5),0)
#ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.imshow(hold_1, cmap="gray", vmin=0, vmax=255)
#plt.imshow(blur)
plt.show()

#Erosion and Dilatation
kernel=np.ones((15,15), np.uint8)
img_dilation = cv2.dilate(hold, kernel, iterations=1)
img_erode = cv2.erode(img_dilation, kernel, iterations=1)

#Remove noise from image
img_erode= cv2.medianBlur(img_erode, 7)

#plt.subplot(222)
plt.title('Dilation + erosion')
plt.imshow(img_erode, cmap='gray', vmin=0, vmax=255)
#plt.show()

#Label image
ret, labels = cv2.connectedComponents(img_erode)
label_hue = np.uint8(179*labels/np.max(labels))
black_ch = 225 * np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, black_ch, black_ch])

labeled_img =cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

labeled_img[label_hue==0]=0



#plt.subplot(222)
plt.title('objects counted:' + str(ret-1))
plt.imshow(labeled_img)

print('objects number is:', ret-1)

plt.show()




