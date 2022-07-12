from turtle import right
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from line import Line
import os

def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold    
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    abs_sobel = np.absolute(sobel)
    
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sbinary

def pipeline(img, s_thresh=(50, 255), l_thresh=(50,255)):
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    equ = cv2.equalizeHist(gray)

    # show the histogram equalized image
    f, (ax1) = plt.subplots(1, 1, figsize=(24, 12))
    f.tight_layout()
    ax1.imshow(gray)
    ax1.set_title('Histogram Equalized', fontsize=50)
    plt.imsave('output_images/examples/example_perspective_hist_eq.jpg', equ)

    #img = cv2.cvtColor(equ,cv2.COLOR_GRAY2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    
    max_l = np.max(l_channel)
        
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    
    s_binary[((s_channel >= 90) & (l_channel >= 100)) | 
             ((s_channel < 32) & (l_channel >= 0.85*max_l))] = 1

    # if my HLS based thresholding can't find many lane line votes, fall back to sobel based approach
    if np.sum(s_binary) <= 600:
        s_binary = np.zeros_like(s_channel)
        tmp_x = abs_sobel_thresh(equ, orient='x', sobel_kernel=15, thresh=(5, 255))
        s_binary = np.copy(tmp_x)
        s_binary = s_binary
    
    # get rid of any results in the lower middle part of the image. Helps with problems made by HOV marker    
    s_binary[gray.shape[0]-300:gray.shape[0], 
                                int(gray.shape[1]/2)-150:int(gray.shape[1]/2)+150] = 0

    color_binary = np.dstack(( s_binary, s_binary, s_binary))

    return color_binary

# Function comes from "Undistort and Transform Perspective" unit in the project's lesson
def corners_unwarp(img, corners):  
    # Convert undistorted image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Choose offset from image corners to plot detected corners
    # This should be chosen to present the result at the proper aspect ratio
    # I went for 150 in the x direction, but 0 in the y direction since my region of interest is at the bottom
    # of the image already
    xoffset = 150 # offset for dst points
    yoffset = 0 # offset for dst points
    # Grab the image shape
    img_size = (gray.shape[1], gray.shape[0])

    # For source points I'm grabbing the outer four detected corners
    src = np.float32([corners[0], corners[1], corners[2], corners[3]])
    
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result 
    # again, not exact, but close enough for our purposes
    dst = np.float32([[xoffset, yoffset], [img_size[0]-xoffset, yoffset], 
                                 [img_size[0]-xoffset, img_size[1]-yoffset], 
                                 [xoffset, img_size[1]-yoffset]])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)

    # Return the resulting image and matrix
    return warped, M

# Function comes from "Undistort and Transform Perspective" unit in the project's lesson
def corners_warp(img, corners):  
    # at this point in the process, image is already just 1 image plane - don't have to cvt to grayscale
    # since rest of fxn depends on "gray" variable, just copy img to gray
    gray = np.copy(img)
    
    # Choose offset from image corners to plot detected corners
    # This should be chosen to present the result at the proper aspect ratio
    # I went for 150 in the x direction, but 0 in the y direction since my region of interest is at the bottom
    # of the image already
    xoffset = 150 # offset for dst points
    yoffset = 0 # offset for dst points
    # Grab the image shape
    img_size = (gray.shape[1], gray.shape[0])

    # For source points I'm grabbing the outer four detected corners
    dst = np.float32([corners[0], corners[1], corners[2], corners[3]])
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result 
    # again, not exact, but close enough for our purposes
    src = np.float32([[xoffset, yoffset], [img_size[0]-xoffset, yoffset], 
                                 [img_size[0]-xoffset, img_size[1]-yoffset], 
                                 [xoffset, img_size[1]-yoffset]])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)

    # Return the resulting image and matrix
    return warped, M

def compute_lane_line_polynomials(top_down, binary_warped, undistort, right_line, left_line):
    # from "Finding the lines"
    # histogram of the bottom of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9

    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 100
    #margin = 50

    # Set minimum number of pixels found to recenter window
    minpix = 100

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        #cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        #(0,255,0), 2) 
        #cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        #(0,255,0), 2) 

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # keeps track of how many points I add for each frame
    left_line.len_per_batch_x.append(len(leftx))
    left_line.len_per_batch_y.append(len(lefty))
    right_line.len_per_batch_x.append(len(rightx))
    right_line.len_per_batch_y.append(len(righty))
   
    # add x and y to running list of all points
    if (len(left_line.allx) < 2):
        left_line.allx = np.copy(leftx)
    else:
        left_line.allx = np.append(left_line.allx, leftx, axis=0)
        
    if (len(right_line.allx) < 2):
        right_line.allx = np.copy(rightx)
    else:
        right_line.allx = np.append(right_line.allx, rightx, axis=0)
        
    if (len(left_line.ally) < 2):
        left_line.ally = np.copy(lefty)
    else:
        left_line.ally = np.append(left_line.ally, lefty, axis=0)
        
    if (len(right_line.ally) < 2):
        right_line.ally = np.copy(righty)
    else:
        right_line.ally = np.append(right_line.ally, righty, axis=0)   
    
    # Fit a second order polynomial to each
    
    # make decisions based on the last 5 frames of data
    n_units = 5
    if (len(left_line.len_per_batch_x) <= n_units):
        go_back_n_leftx = np.sum(left_line.len_per_batch_x)
        go_back_n_lefty = np.sum(left_line.len_per_batch_y)
        go_back_n_rightx = np.sum(right_line.len_per_batch_x)
        go_back_n_righty = np.sum(right_line.len_per_batch_y)
    else:
        go_back_n_leftx = np.sum(left_line.len_per_batch_x[len(left_line.len_per_batch_x)-n_units:])
        go_back_n_lefty = np.sum(left_line.len_per_batch_y[len(left_line.len_per_batch_y)-n_units:])
        go_back_n_rightx = np.sum(right_line.len_per_batch_x[len(right_line.len_per_batch_x)-n_units:])
        go_back_n_righty = np.sum(right_line.len_per_batch_y[len(right_line.len_per_batch_y)-n_units:])
    
    if (go_back_n_leftx == 0 or go_back_n_lefty == 0 or go_back_n_rightx == 0 or go_back_n_rightx == 0):
        left_fit = np.copy(left_line.best_fit)
        right_fit = np.copy(right_line.best_fit)
    else:
        left_fit = np.polyfit(left_line.ally[len(left_line.ally)-go_back_n_lefty:], 
                          left_line.allx[len(left_line.allx)-go_back_n_leftx:], 2)
        right_fit = np.polyfit(right_line.ally[len(right_line.ally)-go_back_n_righty:], 
                          right_line.allx[len(right_line.allx)-go_back_n_rightx:], 2)
    
    # track the best fit
    if len(left_line.best_fit) <= 1:
        left_line.best_fit = np.copy(left_fit)            
    if len(right_line.best_fit) <= 1:
        right_line.best_fit = np.copy(right_fit)    

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # if there's a drastic change in the line coefficients, mostly keep the old best fit
    if np.absolute(left_line.best_fit[1] - left_fit[1]) >= 0.15*np.absolute(left_line.best_fit[1]):
        left_fit = 0.15*left_fit + 0.85*left_line.best_fit
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    if np.absolute(right_line.best_fit[1] - right_fit[1]) >= 0.15*np.absolute(right_line.best_fit[1]):
        right_fit = 0.15*right_fit + 0.85*right_line.best_fit
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                  ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                  ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    
    # overlay the polygon onto the image
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        
    # update the line classes
    left_line.result = np.copy(result)
    right_line.result = np.copy(result)    
    
    left_line.current_fit = np.copy(left_fit)
    right_line.current_fit = np.copy(right_fit)         

    left_line.best_fit = np.copy(left_fit)
    right_line.best_fit = np.copy(right_fit)
    
    right_fit = np.copy(right_line.best_fit)
    left_fit = np.copy(left_line.best_fit)
    
    return left_fitx, right_fitx, lefty, righty, leftx, rightx, ploty, result

def calculate_lane_curvature(lefty, righty, leftx, rightx, ploty):
    # meters per pixel in y dimension
    ym_per_pix = 30/720 
    
    # meters per pixel in x dimension
    xm_per_pix = 3.7/700 

    # calculate line based on input points but keep aware of the pixels/meter conversion
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    # calculate the radius based on the polynomial and the pixels/meter conversion
    left_curverad = ((1 + (2*left_fit_cr[0]*np.max(ploty)*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*np.max(ploty)*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    return left_curverad, right_curverad

def big_pipeline(img_fname, mtx, dist):
    image = mpimg.imread(os.path.realpath(img_fname))
    
    # hard coded for udacity
    corners = np.float32(
        [
            [500, 500],
            [780, 500],
            [1130, 720],
            [150, 720]               
         ])
    
    # hard coded for track
    #corners = np.float32(
    #    [
    #        [0, 400],
    #        [1280, 400],
    #        [1280, 500],
    #        [0, 500]               
    #     ])

    # show the original image
    f, (ax1) = plt.subplots(1, 1, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=50)    
    plt.imsave('output_images/examples/example_orig_' + os.path.basename(img_fname), image)
    
    # undistort the image
    undist = cv2.undistort(image, mtx, dist, None, mtx)

    # show undistorted image
    f, (ax1) = plt.subplots(1, 1, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(undist)
    ax1.set_title('Undistorted Image', fontsize=50)
    plt.imsave('output_images/examples/example_undist_' + os.path.basename(img_fname), undist)

    # show pespective transform input coordinates on top of undistorted image
    plt.figure(figsize=(16,16))
    plt.imshow(undist)
    plt.plot(corners[0][0], corners[0][1], 'x')
    plt.plot(corners[1][0], corners[1][1], 'x')
    plt.plot(corners[2][0], corners[2][1], 'x')
    plt.plot(corners[3][0], corners[3][1], 'x')
    plt.savefig('output_images/examples/example_perspective_corners_' + os.path.basename(img_fname))
    
    # perspective transform
    top_down, perspective_M = corners_unwarp(undist, corners)

    # find the lane line votes
    binary_warped_stacked = pipeline(top_down)    
    
    # show the perspective transform
    f, (ax1) = plt.subplots(1, 1, figsize=(24, 12))
    f.tight_layout()
    ax1.imshow(top_down)
    ax1.set_title('Undistorted and Warped Image', fontsize=50)
    plt.imsave('output_images/examples/example_perspective_unwarped_' + os.path.basename(img_fname), top_down)
    
    # show lane line votes
    f, (ax1) = plt.subplots(1, 1, figsize=(24, 12))
    f.tight_layout()
    ax1.imshow(binary_warped_stacked)
    ax1.set_title('Thresholding', fontsize=50)
    plt.imsave('output_images/examples/example_thresholded_' + os.path.basename(img_fname), binary_warped_stacked)
    
    binary_warped = binary_warped_stacked[:,:,0]

    # example from "Finding the lines" example
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    # compute lane line polynomials
    left_line = Line()
    right_line = Line()
    left_fitx, right_fitx, lefty, righty, leftx, rightx, ploty, result = compute_lane_line_polynomials(top_down, binary_warped, undist, right_line, left_line)
  
    # overlay lane lines
    f, (ax1) = plt.subplots(1, 1, figsize=(24, 12))
    f.tight_layout()
    ax1.imshow(result)
    ax1.plot(left_fitx, ploty, color='yellow')
    ax1.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.savefig('output_images/examples/example_lines_' + os.path.basename(img_fname))

    # get the transform matrix to put the results back in the car camera's perspective
    replacement, Minv = corners_warp(result, corners)

    # calculate the radius
    left_curverad, right_curverad = calculate_lane_curvature(lefty, righty, leftx, rightx, ploty)

    # from "tips and tricks for the project"
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(image[:,:,0]).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    
    # find where the lane lines are in the car camera's perspective
    found_left = 0
    found_right = 0
    for i in range(0, newwarp.shape[1]):
        #print (newwarp[newwarp.shape[0]-1, i, 1])
        if newwarp[newwarp.shape[0]-1, i, 1] > 0 and found_left == 0 and found_right == 0:
            found_left = i
        if found_left > 0 and found_right == 0 and newwarp[newwarp.shape[0]-1, i, 1] == 0:
            found_right = i-1
    
    # calculate the car's offset from the lane center
    lane_midpoint = 0.5*(found_left+found_right)
    car_offset = lane_midpoint - result.shape[1]/2
    car_offset_m = car_offset*3.7/1280 #meters per pixel in x direction
               
    
    # Combine the result polygon with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    comb_result = np.zeros((image.shape[0], image.shape[1]+binary_warped_stacked.shape[1], 3), dtype=np.uint8)
    comb_result[0:result.shape[0], 0:result.shape[1],:] = result
    comb_result[0:binary_warped_stacked.shape[0], result.shape[1]:result.shape[1]+binary_warped_stacked.shape[1],:] = 255*binary_warped_stacked
    comb_result = cv2.putText(img=np.copy(comb_result), text="RoC: %.2f m" % (0.5*(left_curverad+right_curverad)), org=(1600,200),fontFace=3, fontScale=3, color=(0,0,255), thickness=5)
    comb_result = cv2.putText(img=np.copy(comb_result), text="Offset: %.3f m" % (car_offset_m), org=(1600,300),fontFace=3, fontScale=3, color=(0,0,255), thickness=5)
    
    # show the result
    f, (ax1) = plt.subplots(1, 1, figsize=(24, 9))
    ax1.imshow(comb_result)
    ax1.set_title('Resulting Image', fontsize=50)
    ax1.plot(left_fitx+1280, ploty, color='yellow')
    ax1.plot(right_fitx+1280, ploty, color='yellow')
    #plt.xlim(1280, 2560)
    #plt.ylim(720, 0)
    plt.savefig('output_images/' + os.path.basename(img_fname))    
    
    return result