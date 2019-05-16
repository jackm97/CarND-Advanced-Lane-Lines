import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob

def undistort(image):
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
    
    img_size = (image.shape[1],image.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    
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
    
    src = np.float32([[275,670],[1033,670],[573,465],[710,465]])
    dst = np.float32([[380,720],[900,720],[380,0],[900,0]])
    
    if inverse==False:
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(image, M, (1280, 720))
        
    return warped

def color_thresh(image,thresh=(0,255)):
    hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    gray = hls[:,:,2]
    binary_thresh = np.zeros_like(gray)
    binary_thresh[((gray>120) & (hsv[:,:,1]>120)) | ((gray<40) & (hsv[:,:,1]>120))] = 1
    
    return gray,binary_thresh

def sobelx_thresh(image,thresh=(0,255),ksize=5):
    
    gradx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=ksize)
    gradx_abs = np.abs(gradx)
    gradx_abs = np.uint8(255*gradx_abs/np.max(gradx_abs))
    
    binary_thresh = np.zeros_like(gradx_abs)
    binary_thresh[(gradx_abs>=thresh[0]) & (gradx_abs<=thresh[1])] = 1
    
    return gradx, binary_thresh

def sobely_thresh(image,thresh=(0,255),ksize=5):
    
    grady = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=ksize)
    grady_abs = np.abs(gradx)
    grady_abs = np.uint8(255*gradx_abs/np.max(gradx_abs))
    
    binary_thresh = np.zeros_like(grady_abs)
    binary_thresh[(grady_abs>=thresh[0]) & (grady_abs<=thresh[1])] = 1
    
    return grady, binary_thresh

def sobelxy_thresh(gradx,grady,thresh=(0,255)):
    
    gradxy = np.sqrt(gradx**2 + grady**2)
    gradxy = np.uint8(255*gradxy/np.max(gradxy))
    
    binary_thresh = np.zeros_like(gradxy)
    binary_thresh[(gradxy>=thresh[0]) & (gradxy<=thresh[1])] = 1
    
    return gradxy,binary_thresh

def sobeldir_thresh(gradx,grady,thresh=(0,255)):
    
    graddir = np.arctan2(grady,gradx)
    graddir = np.uint8(255*graddir/np.max(graddir))
    
    binary_thresh = np.zeros_like(graddir)
    binary_thresh[(graddir>=thresh[0]) & (graddir<=thresh[1])] = 1
    
    return graddir, binary_thresh


    
    
        
        
        