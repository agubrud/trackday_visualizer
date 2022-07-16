import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

def print_img(img):
    f, (ax1) = plt.subplots(1, 1, figsize=(24, 12))
    f.tight_layout()
    ax1.imshow(img)

def save_image(img, title, outputFile):
    f, (ax1) = plt.subplots(1, 1, figsize=(24, 12))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title(title, fontsize=50)
    plt.imsave(outputFile, img)

def save_figure(title, image, lines, anotherLine, lineOffset, outputFile):
    f, (ax1) = plt.subplots(1, 1, figsize=(24, 9))
    ax1.imshow(image)
    if title is not None:
        ax1.set_title(title, fontsize=50)
    for l in lines:
        ax1.plot(l+lineOffset, anotherLine, color='yellow')
    plt.savefig(outputFile)    

def save_corner_overlay(img, outputFile, corners):
    plt.figure(figsize=(16,16))
    plt.imshow(img)
    plt.plot(corners[0][0], corners[0][1], 'x')
    plt.plot(corners[1][0], corners[1][1], 'x')
    plt.plot(corners[2][0], corners[2][1], 'x')
    plt.plot(corners[3][0], corners[3][1], 'x')
    plt.savefig(outputFile)