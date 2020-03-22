import cv2
import numpy as np
import sys
from PIL import Image, ImageFilter


def mask_by_color( bgr, im ):
    # read the image
    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    thresh = 85
    lower_yellow = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
    upper_yellow = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])

    # select just one color of the image
    mask = cv2.inRange(im,lower_yellow, upper_yellow)

    # bitwise-and mask and original image
    res = cv2.bitwise_and(im,im, mask= mask)
    shape = im.shape
    imbw = np.zeros(shape)
    imbw.fill(255)
    res3 = cv2.bitwise_and(imbw,imbw,mask = mask)
    return res3

def apply_effects(mask,dimens):
    resblur = cv2.GaussianBlur(mask,(15,15),0)
    ret,thresh1 = cv2.threshold(resblur,127,255,cv2.THRESH_BINARY)
    resized = cv2.resize(thresh1,dimens)
    resized = resized.astype(np.float32)
    return resized

def mask2p(m):
    img = cv2.cvtColor(m, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img.astype(np.uint8)).convert("L")
    return im_pil

def create_mask_by_color(rgb_color,map_image):
    shape = map_img.shape
    maskedbycolor = mask_by_color(rgb_color,map_img)
    maskedwitheffects = apply_effects(maskedbycolor,(shape[1],shape[0]))
    maskedimage = mask2p(maskedwitheffects)
    return maskedimage


# # read the image
map_img = cv2.imread("img/map.png")

# define range of yellow in rgb
roads_color = [4,255,253]
terrain_color = [255,0,253]
bush_color = [0,255,30]

masked_terrain =  create_mask_by_color(roads_color, map_img)
masked_water =  create_mask_by_color(terrain_color, map_img)
masked_bush =  create_mask_by_color(bush_color, map_img)

texture_terrain = Image.open('img/terrain.png').convert("RGB")
texture_water = Image.open('img/water.png').convert("RGB")
texture_bush = Image.open('img/bush.png').convert("RGB")

out = Image.new('RGB', (1138, 496),(255,255,255))
out.paste(texture_terrain,(0,0),masked_terrain)

out.paste(texture_water,(0,0),masked_water)

out.paste(texture_bush,(0,0),masked_bush)
out.show()

