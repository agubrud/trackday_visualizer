import os
import cv2
import numpy as np

def calibrate():# process calibration images
    img_list =os.listdir("./camera_cal/")

    # objpoints are for the nx*ny grid
    # imgpoints are where the chessboard corners lie in the calibration image pixels
    objpoints = []
    imgpoints = []

    # iterate through all of the calibration images
    for i in range(len(img_list)):
        image = cv2.imread('./camera_cal/' + img_list[i])
        
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