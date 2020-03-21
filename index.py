import cv2
import numpy as np
import sys
from PIL import Image, ImageFilter
print(sys.version)
# read the image
im = cv2.imread("img/map.png")

img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im_pil = Image.fromarray(img)

# convert bgr to hsv
# hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)

# define range of yellow in rgb
bgr = [4,255,253]
thresh = 55

lower_yellow = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
upper_yellow = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])

# select just one color of the image
mask = cv2.inRange(im,lower_yellow, upper_yellow)

# bitwise-and mask and original image
res = cv2.bitwise_and(im,im, mask= mask)

res2 = cv2.bitwise_not(im,im, mask= mask)

# cv2.imshow("Res",res)
# cv2.waitKey()

# cv2.imshow("Res 2",res2)
# cv2.waitKey()

shape = im.shape
imbw = np.zeros(shape)
imbw.fill(255)
res3 = cv2.bitwise_not(imbw,imbw,mask = mask)
# cv2.imshow("Res3",res3)
# cv2.waitKey()

resblur = cv2.GaussianBlur(res3,(15,15),0)
# cv2.imshow("Resblur",resblur)
# cv2.waitKey()

ret,thresh1 = cv2.threshold(resblur,127,255,cv2.THRESH_BINARY)
# cv2.imshow('thresh',thresh1)
# cv2.waitKey()

resized = cv2.resize(thresh1,(shape[1],shape[0]))
resized = resized.astype(np.float32)

img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

im_pil = Image.fromarray(img.astype(np.uint8)).convert("L")
print(im_pil.mode)

texture = Image.open('img/terrain.png').convert("RGB")
print(texture.mode)

out = Image.new('RGBA', (1138, 496))
out.paste(texture,(0,0),im_pil)
out.show()
