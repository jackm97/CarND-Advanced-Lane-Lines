import numpy as np
import cv2
import glob

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

def undistort(image,mtx,dist):
    undistorted_image = cv2.undistort(image,mtx,dist,None,mtx)
    
    return undistorted_image

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

def sobelx_thresh(image,thresh=(0,255),ksize=5):
    """
    Performs sobel in x direction and thresh
    """
    
    gradx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=ksize)
    gradx_abs = np.abs(gradx)
    gradx_abs = np.uint8(255*gradx_abs/np.max(gradx_abs))
    
    binary_thresh = np.zeros_like(gradx)
    binary_thresh[(gradx_abs>=thresh[0]) & (gradx_abs<=thresh[1])] = 1
    
    return gradx_abs, binary_thresh

def sobely_thresh(image,thresh=(0,255),ksize=5):
    """
    Performs sobel in y direction and thresh
    """
    
    grady = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=ksize)
    grady_abs = np.abs(grady)
    grady_abs = np.uint8(255*grady_abs/np.max(grady_abs))
    
    binary_thresh = np.zeros_like(grady_abs)
    binary_thresh[(grady_abs>=thresh[0]) & (grady_abs<=thresh[1])] = 1
    
    return grady_abs, binary_thresh

def sobelxy_thresh(gradx,grady,thresh=(0,255)):
    """
    Returns the magnitude of sobelx and sobely scaled for a peak of 255
    """
    
    gradxy = np.sqrt(gradx**2 + grady**2)
    gradxy = np.uint8(255*gradxy/np.max(gradxy))
    
    binary_thresh = np.zeros_like(gradxy)
    binary_thresh[(gradxy>=thresh[0]) & (gradxy<=thresh[1])] = 1
    
    return gradxy,binary_thresh

def sobeldir_thresh(gradx,grady,thresh=(0,255)):
    """
    Returns direction of gradient scaled from [0,255]
    """
    
    graddir = np.arctan2(grady,gradx)
    graddir = np.uint8(255*graddir/np.max(graddir))
    
    binary_thresh = np.zeros_like(graddir)
    binary_thresh[(graddir>=thresh[0]) & (graddir<=thresh[1])] = 1
    
    return graddir, binary_thresh

def get_warped_binary(image,mtx,dist):
    dst = undistort(image,mtx,dist)
    
    hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
    yellow_thresh = np.zeros_like(hsv[:,:,1])
    yellow_thresh[((hsv[:,:,0] > 15) & (hsv[:,:,0] < 30)) & (hsv[:,:,1] > 100) & (hsv[:,:,2]>50)] = 1
    
    white_thresh = np.zeros_like(yellow_thresh)
    white_thresh[(hsv[:,:,1] < 30) & (hsv[:,:,2]>200)] = 1
    
    color_thresh = (white_thresh + yellow_thresh).astype(np.uint8)
    
    hls = cv2.cvtColor(dst, cv2.COLOR_BGR2HLS)
    gradx, gradx_thresh = sobelx_thresh(hls[:,:,2],thresh=(30,120),ksize=9)
    
    binary = np.bitwise_or(color_thresh,gradx_thresh.astype(np.uint8))
    
    warped = warp(binary)
    
    return warped, dst
    
    
