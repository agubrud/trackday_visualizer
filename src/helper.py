import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

def print_img(img):
    f, (ax1) = plt.subplots(1, 1, figsize=(24, 12))
    f.tight_layout()
    ax1.imshow(img)

def save_figure(img, title, outputFile):
    f, (ax1) = plt.subplots(1, 1, figsize=(24, 12))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title(title, fontsize=50)
    plt.imsave(outputFile, img)