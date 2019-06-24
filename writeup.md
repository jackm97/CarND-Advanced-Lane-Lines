## Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibrate.jpg "Undistorted"
[image2]: ./output_images/test5_calibrate.jpg "Road Transformed"
[image3]: ./output_images/thresh.jpg "Binary Example"
[image4]: ./output_images/warped_binary.jpg "Warp Example"
[image5]: ./output_images/poly_fit.jpg "Fit Visual"
[image6]: ./output_images/final.jpg "Output"
[video1]: ./test_videos_output/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. How I computed the camera matrix and distortion coefficients.

The code for this step is located in lines 14 through 43 of the file called `img_manip.py` 
```python
def get_calibration_matrix():
    """
    Parameter:
        image - image to be undistorted
    Returns:
        undistorted_image
    """
    
    objp = np.zeros((9*6,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
    
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    images = glob.glob('./camera_cal/*.jpg')
    
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
    
    img_size = (1280,720)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    
    return mtx,dist
```

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. An example of a distortion-corrected image.

Using the distortion matrix and distortion coeffecients from the calibration step and I apply the `cv2.undistort()`. An example result is shown below:
![alt text][image2]

#### 2. How I used color transforms and gradients to create a thresholded binary image

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 104 through 116 in `img_manip.py`):
```python    
    hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
    yellow_thresh = np.zeros_like(hsv[:,:,1])
    yellow_thresh[((hsv[:,:,0] > 15) & (hsv[:,:,0] < 30)) & (hsv[:,:,1] > 100) & (hsv[:,:,2]>50)] = 1

    white_thresh = np.zeros_like(yellow_thresh)
    white_thresh[(hsv[:,:,1] < 30) & (hsv[:,:,2]>200)] = 1

    color_thresh = (white_thresh + yellow_thresh).astype(np.uint8)

    hls = cv2.cvtColor(dst, cv2.COLOR_BGR2HLS)
    gradx, gradx_thresh = sobelx_thresh(hls[:,:,2],thresh=(30,120),ksize=9)

    binary = np.bitwise_or(color_thresh,gradx_thresh.astype(np.uint8))
```
Here's an example of my output for this step:

![alt text][image3]

#### 3. How I peformed a perspective transform.

The code for my perspective transform includes a function called `warp()`, which appears in lines 50 through 74 in the file `img_manip.py`.  The `warp()` function takes as inputs an image (`image`), (`inverse`) for handling whether or not we're warping or un warping the image, as well as source (`src`) and destination (`dst`) points.  I chose to hardcode the source and destination points.

```python
def warp(image,inverse=False,src=None,dst=None):
    """
    Warps the image to top-down view
    
    Parameters:
        image - image to be warped (should be undistorted)
        inverse - if set to true, transforms from warped to unwarped (default:False)
        src - Vertices in the source image to be mapped to destination image (leave as None if using the default values)
        dst - Vertices in the dst image that the src points are mapped to (leave as None if using the default values)
    Returns:
        warped
    """
    
    if src==None and dst==None:
        src = np.float32([[275,700],[1033,700],[573,465],[710,465]])
        dst = np.float32([[300,720],[980,720],[300,0],[980,0]])
    
    if inverse==False:
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(image, M, (1280, 720))
    else:
        M = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(image, M, (1280, 720))
        
    return warped
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. How I identified lane line pixels and fit their positions to a polynomial.

I found lane pixels using either the sliding histogram method or by searching around a pre-defined polynomial fit. The pre-defined polynomial fit is faster than the sliding historgram method and is used if, in the video pipeline, the previous frame found a reasonable polynomial fit for the lane lines. The code for this is found from lines 68 to 160 in `lane_finder.py`.

A second-order polynomial was then fit to the y and x positions of the pixels found above.

Lane lines and a polynomial are overlayed (code in `lane_finder.py` from lines 249 to 267):

![alt text][image5]

#### 5. How I calculated the radius of curvature of the lane and the position of the car relative to the center of the lane.

I did this in lines 269 through 264 in my code in `lane_finder.py`

```python
    def calc_curv(self,colored_lanes):
        
        left_fit = self.left_line.best_fit
        right_fit = self.right_line.best_fit
        my = 3.0/150
        mx = 3.7/750

        A_left = mx/(my**2)*left_fit[0]
        B_left = mx/(my**2)*left_fit[1]
        A_right = mx/(my**2)*right_fit[0]
        B_right = mx/(my**2)*right_fit[1]
        y = colored_lanes.shape[0]

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
        self.left_line.line_base_pos = mx*(leftx_base - midpoint)
        self.right_line.line_base_pos = mx*(rightx_base - midpoint)
```

#### 6. An example of the final product.
Finally the overlay is unwarped and overlayed with the original image. An example is shown below:

![alt text][image6]

---

### Pipeline (video)

#### 1. An example of the video pipeline.

An example usage of the video pipeline is in cell 7 of the [Ipython notebook](./"Advanced Lane Detection.ipynb") along with the resulting video.

```python
import lane_finder as lf
output = 'test_videos_output/project_video.mp4'
finder = lf.lane_finder()
%time finder.process_video("project_video.mp4",output)
```

Here's a [link to my video result](./test_videos_output/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In order for my pipeline to work it needs to work in varying light and road conditions. For this I used the HSV and HLS colorspaces for thresholding. The `lane_finder` class handles the processing of videos which includes averaging over frames and checking for failed frames. Although this pipeline works on the project video, it doesn't work well with the challenge videos. Some steps to make it more robust include adaptive thresholding, less hardcoding, and a better method for handling failed frames. Essentially, the pipeline fails in extremely winding conditions and poor road conditions (i.e. black tar lines, sharp shadows, etc.)  
