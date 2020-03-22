
def mask_by_color( color,img_path ):
    # read the image
    im = cv2.imread(img_path)

    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    thresh = 55
    lower_yellow = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
    upper_yellow = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])

    # select just one color of the image
    mask = cv2.inRange(im,lower_yellow, upper_yellow)

    # bitwise-and mask and original image
    res = cv2.bitwise_and(im,im, mask= mask)

    shape = im.shape
    imbw = np.zeros(shape)
    imbw.fill(255)
    res3 = cv2.bitwise_not(imbw,imbw,mask = mask)
    return res3