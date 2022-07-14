from ast import arg
from calibrate import calibrate
from proc import big_pipeline
import argparse
import pickle as pkl
import os

def main():
    if not os.path.exists('./output_images/'):
        os.makedirs('./output_images/')
    if not os.path.exists('./output_images/examples/'):
        os.makedirs('./output_images/examples')

    mtx, dist = calibrate(calibrationMethod=args.calibrationMethod, calibrationDir=args.calibrationDir, calibrationPkl=args.calibrationPkl)

    big_pipeline(mtx, dist, inputImage=args.inputImage)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputImage", required=True, help="Input image", dest="inputImage")
    parser.add_argument("-cd", "--calibrationDir", required=False, default="./camera_cal/", help="Path to calibration images directory", dest="calibrationDir")
    parser.add_argument("-cp", "--calibrationPkl", required=False, default="./camera_cal/camera_cal.pkl", help="Path to precomputed calibration pickle", dest="calibrationPkl")
    parser.add_argument("-cm", "--calibrationMethod", required=False, default="pickle", choices=["directory", "pickle"], help="Mode to choose on-demand calibration or use preexisting pickle", dest="calibrationMethod")
    args = parser.parse_args()
    main()
