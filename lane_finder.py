"""
lane_finder.py

Use the lane_finder class to process driving footage and find the lane lines

Usage:
    import lane_finder as lf
    finder = lf.lane_finder()
    finder.process_video(infile, outfile)
        infile - filename of input footage
        outfile - filename of output processed footage
        
JACK MYERS 19 JUNE 2019
"""

from moviepy.editor import VideoFileClip
from IPython.display import HTML
import img_manip
import cv2
import numpy as np

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recentx_fitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = np.array([False])  
        #polynomial coefficients for the most recent fit
        self.current_fit = np.array([False])  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

# define a class to process the video
class lane_finder:
    
    def __init__(self,mtx=None,dist=None):
        if (mtx==None and dist==None):
            self.mtx, self.dist = img_manip.get_calibration_matrix()
        else:
            self.mtx=mtx
            self.dist=dist
    
        self.left_line = Line()
        self.right_line = Line()
    
    
    def process_video(self,infile, outfile):
        self.left_line = Line()
        self.right_line = Line()
        clip = VideoFileClip(infile)
        processed_clip = clip.fl_image(lambda image:self.process_frame(image))#NOTE: this function expects color images!!
        processed_clip.write_videofile(outfile, audio=False)
        clip.close()
        
    def find_lane_pixels(self,binary_warped,nwindows=9,margin=50,minpix=100):
        
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

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
        if len(nonzerox[left_lane_inds]) > 0:
            self.left_line.allx = nonzerox[left_lane_inds]
            self.left_line.ally = nonzeroy[left_lane_inds] 
        if len(nonzerox[right_lane_inds]) > 0:
            self.right_line.allx = nonzerox[right_lane_inds]
            self.right_line.ally = nonzeroy[right_lane_inds]

        return out_img

    def search_around_poly(self,binary_warped, margin=50):
        
        left_fit = self.left_line.best_fit
        right_fit = self.right_line.best_fit

        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                        left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                        left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                        right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                        right_fit[1]*nonzeroy + right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        if len(nonzerox[left_lane_inds]) > 0:
            self.left_line.allx = nonzerox[left_lane_inds]
            self.left_line.ally = nonzeroy[left_lane_inds] 
        if len(nonzerox[right_lane_inds]) > 0:
            self.right_line.allx = nonzerox[right_lane_inds]
            self.right_line.ally = nonzeroy[right_lane_inds]

        return out_img

    def fit_polynomial(self,binary_warped):
        

        # Find our lane pixels first using search ahead or sliding window depending on
        # if the previous frame was detected or not
        if (self.left_line.detected and self.right_line.detected):
            out_img = self.search_around_poly(binary_warped)
        else:
            out_img = self.find_lane_pixels(binary_warped)
        leftx = self.left_line.allx
        lefty = self.left_line.ally
        rightx = self.right_line.allx
        righty = self.right_line.ally

        # Fit a second order polynomial for the current fit
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        self.left_line.current_fit = left_fit
        self.right_line.current_fit = right_fit


        # Get x values for current fit
        y_values = np.linspace(0,binary_warped.shape[0]-1,binary_warped.shape[0])
        if (self.left_line.best_fit[0]!=False):

            self.left_line.diffs = np.abs(self.left_line.current_fit - self.left_line.best_fit)

            if (np.linalg.norm(self.left_line.diffs)/np.linalg.norm(self.left_line.best_fit) < .20):
                self.left_line.detected = True
                self.left_line.recentx_fitted.append(self.left_line.current_fit[0]*y_values**2+
                                                self.left_line.current_fit[1]*y_values+
                                                self.left_line.current_fit[2])
                if (len(self.left_line.recentx_fitted)>3):
                    self.left_line.recentx_fitted = self.left_line.recentx_fitted[1:]
                self.left_line.bestx = np.mean(self.left_line.recentx_fitted,axis=0)
            else:
                self.left_line.detected = False
                self.left_line.recentx_fitted.append(self.left_line.recentx_fitted[-1])
                if (len(self.left_line.recentx_fitted)>3):
                    self.left_line.recentx_fitted = self.left_line.recentx_fitted[1:]
                self.left_line.bestx = np.mean(self.left_line.recentx_fitted,axis=0)    
        else:
            self.left_line.detected=True
            self.left_line.recentx_fitted.append(self.left_line.current_fit[0]*y_values**2+
                                                self.left_line.current_fit[1]*y_values+
                                                self.left_line.current_fit[2])
            self.left_line.bestx = np.mean(self.left_line.recentx_fitted,axis=0)



        if (self.right_line.best_fit[0]!=False):

            self.right_line.diffs = np.abs(self.right_line.current_fit - self.right_line.best_fit)

            if (np.linalg.norm(self.right_line.diffs)/np.linalg.norm(self.right_line.best_fit) < .05):
                self.right_line.detected = True
                self.right_line.recentx_fitted.append(self.right_line.current_fit[0]*y_values**2+
                                                self.right_line.current_fit[1]*y_values+
                                                self.right_line.current_fit[2])
                if (len(self.right_line.recentx_fitted)>3):
                    self.right_line.recentx_fitted = self.right_line.recentx_fitted[1:]
                self.right_line.bestx = np.mean(self.right_line.recentx_fitted,axis=0)
            else:
                self.right_line.detected = False
                self.right_line.recentx_fitted.append(self.right_line.recentx_fitted[-1])
                if (len(self.right_line.recentx_fitted)>3):
                    self.right_line.recentx_fitted = self.right_line.recentx_fitted[1:]
                self.right_line.bestx = np.mean(self.right_line.recentx_fitted,axis=0)
        else:
            self.right_line.detected=True
            self.right_line.recentx_fitted.append(self.right_line.current_fit[0]*y_values**2+
                                                self.right_line.current_fit[1]*y_values+
                                                self.right_line.current_fit[2])
            self.right_line.bestx = np.mean(self.right_line.recentx_fitted,axis=0)

        # Get the best fit from the average of x and y values over the last three frames
        self.left_line.best_fit = np.polyfit(y_values,self.left_line.bestx,2)
        self.right_line.best_fit = np.polyfit(y_values,self.right_line.bestx,2)


        ## Visualization ##
        # Colors in the left and right lane regions
        out_img[self.left_line.ally, self.left_line.allx] = [255, 0, 0]
        out_img[self.right_line.ally, self.right_line.allx] = [0, 0, 255]

        return out_img

    def overlay_lane(self,colored_lanes):
        
        left_fit = self.left_line.best_fit
        right_fit = self.right_line.best_fit

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

    def calc_curv(self,colored_lanes):
        
        left_fit = self.left_line.best_fit
        right_fit = self.right_line.best_fit
        my = 3.0/150
        mx = 3.7/900

        A_left = mx/(my**2)*left_fit[0]
        B_left = mx/(my**2)*left_fit[1]
        A_right = mx/(my**2)*right_fit[0]
        B_right = mx/(my**2)*right_fit[1]
        y = colored_lanes.shape[0]//2

        left_curv = (1+(2*A_left*y+B_left)**2)**1.5/np.abs(2*A_left)
        right_curv = (1+(2*A_right*y+B_right)**2)**1.5/np.abs(2*A_right)
        self.left_line.radius_of_curvature = left_curv
        self.right_line.radius_of_curvature = right_curv

        y = np.linspace(0, colored_lanes.shape[0]-1, colored_lanes.shape[0] )
        leftx = left_fit[0]*y**2 + left_fit[1]*y + left_fit[2]
        rightx = right_fit[0]*y**2 + right_fit[1]*y + right_fit[2]
        midpoint = colored_lanes.shape[1]//2
        leftx_base = leftx[-1]
        rightx_base = rightx[-1]
        self.left_line.line_base_pos = mx*(leftx - midpoint)
        self.right_line.line_base_pos = mx*(rightx - midpoint)


    def process_frame(self,image):
        

        # threshold image and return warped top down perspective
        binary_warped, dst = img_manip.get_warped_binary(image,self.mtx,self.dist) 

        # draw lane lines and overlay green lane area
        colored_lanes = self.fit_polynomial(binary_warped)
        colored_lanes = self.overlay_lane(colored_lanes)

        # unwarp
        colored_lanes_unwarped = img_manip.warp(colored_lanes,inverse=True)

        # calculate radii of curvatures and offset from center of lane
        self.calc_curv(colored_lanes)
        left_curv = self.left_line.radius_of_curvature
        right_curv = self.right_line.radius_of_curvature 
        scaled_offset = -1*(self.right_line.line_base_pos + self.left_line.line_base_pos)

        # write information on image
        out_image = cv2.addWeighted(dst,.7,colored_lanes_unwarped,0.8,0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(out_image,"left curvature: " + str(round(left_curv,3)) + "m",(10,50), font, 1,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(out_image,"right curvature: " + str(round(right_curv,3)) + "m",(10,100), font, 1,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(out_image,"offset from center: " + str(round(scaled_offset[0],3)) + "m",(10,150), font, 1,(255,255,255),2,cv2.LINE_AA)

        return out_image
    
    