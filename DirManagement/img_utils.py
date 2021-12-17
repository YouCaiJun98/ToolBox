import cv2
from skimage import img_as_ubyte

def save_img(filepath, img):
    img = img_as_ubyte(img)
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
