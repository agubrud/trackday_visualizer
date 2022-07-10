import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

def print_img(img):
    f, (ax1) = plt.subplots(1, 1, figsize=(24, 12))
    f.tight_layout()
    ax1.imshow(img)