from calibrate import calibrate
import cv2
import matplotlib.pyplot as plt
from proc import big_pipeline
import os

def main():
    mtx, dist = calibrate()
    if not os.path.exists('./output_images/'):
        os.makedirs('./output_images/')
        os.makedirs('./output_images/examples')
    image = cv2.imread('./camera_cal/calibration1.jpg')
    undist = cv2.undistort(image, mtx, dist, None, mtx)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(undist)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig('output_images/calibration1.jpg')

    big_pipeline('miata2.jpg', mtx, dist)
    return

if __name__ == "__main__":
    main()
