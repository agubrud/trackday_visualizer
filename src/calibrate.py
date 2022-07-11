import os
import cv2
import numpy as np
import pickle as pkl
import glob

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
    if kwargs.get("calibrationMethod") == "pickle":
        with open(kwargs.get('calibrationPkl'), 'rb') as handle:
            cameraCalDict = pkl.load(handle)
        return cameraCalDict.values()
    else:
        return _calibrate(calibrationDir=kwargs.get("calibrationDir"))    