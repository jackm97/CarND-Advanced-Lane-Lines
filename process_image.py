import img_manip
import numpy as np
import cv2

def find_lane_pixels(binary_warped,nwindows=9,margin=100,minpix=50):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    offset = leftx_base + (rightx_base-leftx_base)/2 - midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

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
        
        # Identify the nonzero pixels in x and y within the window #
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

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return offset, leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped,sliding_hist=True):
    # Find our lane pixels first
    offset, leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    return out_img, left_fit, right_fit, offset

def overlay_lane(colored_lanes,left_fit,right_fit):
    
    y = np.linspace(0, colored_lanes.shape[0]-1, colored_lanes.shape[0] )
    leftx = left_fit[0]*y**2 + left_fit[1]*y + left_fit[2]
    rightx = right_fit[0]*y**2 + right_fit[1]*y + right_fit[2]
    
    left = np.array([leftx,y]).T
    right =  np.array([rightx,y]).T
    pts = np.int32(np.concatenate([left,right[::-1]]))
    
    poly = np.zeros_like(colored_lanes)
    cv2.fillConvexPoly(poly,pts,(0,255,0))
    
    out_image = cv2.addWeighted(colored_lanes,1,poly,0.5,0)
    
    return out_image

def calc_curv(left_fit, right_fit, offset):
    my = 3.0/150
    mx = 3.7/740
    
    A_left = mx/(my**2)*left_fit[0]
    B_left = mx/(my**2)*left_fit[1]
    A_right = mx/(my**2)*right_fit[0]
    B_right = mx/(my**2)*right_fit[1]
    y = my*600
    
    left_curv = (1+(2*A_left*y+B_left)**2)**1.5/np.abs(2*A_left)
    right_curv = (1+(2*A_right*y+B_right)**2)**1.5/np.abs(2*A_right)
    scaled_offset = mx*offset
    
    return left_curv, right_curv, scaled_offset

def process_image(image,mtx,dist):

    binary_warped, dst = img_manip.get_warped_binary(image,mtx,dist)
    
    colored_lanes, left_fit, right_fit, offset = fit_polynomial(binary_warped)
    colored_lanes = overlay_lane(colored_lanes,left_fit,right_fit)
    colored_lanes_unwarped = img_manip.warp(colored_lanes,inverse=True)
    
    left_curv, right_curv, scaled_offset = calc_curv(left_fit,right_fit,offset)
    
    out_image = cv2.addWeighted(dst,.7,colored_lanes_unwarped,0.8,0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(out_image,"left curvature: " + str(round(left_curv,3)) + "m",(10,50), font, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(out_image,"right curvature: " + str(round(right_curv,3)) + "m",(10,100), font, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(out_image,"offset from center: " + str(round(scaled_offset,3)) + "m",(10,150), font, 1,(255,255,255),2,cv2.LINE_AA)
    
    return out_image 