import os
import cv2
import numpy as np
import pickle as pkl
import glob
import matplotlib.pyplot as plt

def _calibrate(**kwargs):# process calibration images
    img_list =glob.glob(os.path.realpath(f"{kwargs.get('calibrationDir')}/*.jpg"))

    # objpoints are for the nx*ny grid
    # imgpoints are where the chessboard corners lie in the calibration image pixels
    objpoints = []
    imgpoints = []

    # iterate through all of the calibration images
    for img in img_list:
        image = cv2.imread(img)
        
        # the number of inside corners in x
        nx = 9 
        # the number of inside corners in y
        ny = 6

        # extract the chessboard grid and their corresponding mapping to the calibration image pixel
        # space with the chessboard_data function from the previous cell
            
        objp = np.zeros((ny*nx,3), np.float32)
        objp [:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, draw corners
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
        else:
            print ("\tCHESSBOARD NOT FOUND! SKIPPING...")
        
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image.shape[:-1], None, None)

    return mtx, dist

# TODO: Better error handling
def calibrate(**kwargs):
    mtx, dist = None, None
    if kwargs.get("calibrationMethod") == "pickle":
        with open(kwargs.get('calibrationPkl'), 'rb') as handle:
            cameraCalDict = pkl.load(handle)
        mtx, dist = cameraCalDict.values()
    else:
        mtx, dist = _calibrate(calibrationDir=kwargs.get("calibrationDir"))  

    firstCalibrationFile = sorted(glob.glob(f"{kwargs.get('calibrationDir')}/*.jpg"))[0]
    image = cv2.imread(firstCalibrationFile)
    undist = cv2.undistort(image, mtx, dist, None, mtx)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(undist)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig(f"output_images/{os.path.basename(firstCalibrationFile)}")  

    return mtx, dist