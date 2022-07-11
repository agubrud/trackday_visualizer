from ast import arg
from calibrate import calibrate
import cv2
import matplotlib.pyplot as plt
from proc import big_pipeline
import os
import argparse
import pickle as pkl

def main():
    mtx, dist = calibrate(calibrationMethod=args.calibrationMethod, calibrationDir=args.calibrationDir, calibrationPkl=args.calibrationPkl)

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

    big_pipeline(args.inputImage, mtx, dist)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputImage", required=True, help="Input image", dest="inputImage")
    parser.add_argument("-cd", "--calibrationDir", required=False, default="./camera_cal/", help="Path to calibration images directory", dest="calibrationDir")
    parser.add_argument("-cp", "--calibrationPkl", required=False, default="./camera_cal/camera_cal.pkl", help="Path to precomputed calibration pickle", dest="calibrationPkl")
    parser.add_argument("-cm", "--calibrationMethod", required=False, default="pickle", choices=["directory", "pickle"], help="Mode to choose on-demand calibration or use preexisting pickle", dest="calibrationMethod")
    args = parser.parse_args()
    main()
