import cv2 
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import warnings

def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)
    
def window(img, WL=50, WW=350):
    upper, lower = WL + WW//2, WL - WW//2 #225, -125
    X = np.clip(img.copy(), lower, upper)
    return X

def load_dicom_img_three_window_levels(dcm_file):
    # return: 3 slice with 3 windows
    imgs = []
    ds = pydicom.read_file(dcm_file)
    imgx = ds.pixel_array.copy().astype(float)
    
    try:
        imgx = imgx * ds.RescaleSlope
    except:
        print('No RecaleSlop tag!') 
    try:
        imgx = imgx + ds.RescaleIntercept
    except:
        print('No RecaleSlop tag!')
        

    window_sizes = [(40, 80), (80, 200), (600, 2800)] # 3 windows for brain, subdural, bone

    for ws in window_sizes:
        window_center, window_width = ws
        img = window(imgx, WL=window_center, WW=window_width)
        img = img - img.min()
        img = img / img.max()
        img = (img*255).astype(np.uint8)
        imgs.append(img)
    img = np.stack(imgs, -1) # w, h, c
    return img    


def load_dicom_img_three_window_levels_1slice_3windows(dcm_file):
    # return: 1 slice with 3 windows
    imgs = []
    ds = pydicom.read_file(dcm_file)
    img = ds.pixel_array.copy().astype(float)

    try:
        img = img * ds.RescaleSlope
    except:
        print('No RecaleSlop tag!') 
    try:
        img = img + ds.RescaleIntercept
    except:
        print('No RecaleSlop tag!')

    window_sizes = [(40, 80), (80, 200), (600, 2800)] # 3 windows for brain, subdural, bone
    imgs = []
    for ws in window_sizes:
    #     print(ws)
        window_center, window_width = ws
        img2 = window(img, WL=window_center, WW=window_width)
        img2 = img2 - img2.min()
        img2 = img2 / img2.max()
        img2 = (img2 * 255).astype(np.uint8)
        imgs.append(img2)
        
    img3 = np.array(imgs)
    img3 = img3.transpose(1, 2, 0) # w, h, c
    return np.mean(img3, axis=2)

def load_dicom_ncct(dcm_file):
    ds = pydicom.read_file(dcm_file)
    img = ds.pixel_array.copy().astype(float)
    
    try:
        img = img * ds.RescaleSlope
    except:
        print('No RecaleSlop tag!') 
    try:
        img = img + ds.RescaleIntercept
    except:
        print('No RecaleSlop tag!')
        
    return img    

def load_dicom_img(dcm_file):

    ds = pydicom.read_file(dcm_file)
    img = ds.pixel_array.copy().astype(float)
    
    try:
        img = img * ds.RescaleSlope
    except:
        print('No RecaleSlop tag!') 
    try:
        img = img + ds.RescaleIntercept
    except:
        print('No RecaleSlop tag!')
    try:
        WL, WW = [get_first_of_dicom_field_as_int(x) for x in [ds.WindowCenter, ds.WindowWidth]]
        img = window(img, WL=WL, WW=WW)
    except:
        warnings.warn('No WindowCenter or WindowWidth tag')
    
    img -= img.min()
    img /= img.max()
    if len(img.shape) == 2:
        img = np.stack([img]*3, -1)
    img = (img*255).astype(np.uint8)
    return img

def load_dicom_ctp(dcm_file):

    ds = pydicom.read_file(dcm_file)
    img = ds.pixel_array.copy().astype(float)
    
    try:
        img = img * ds.RescaleSlope
    except:
        print('No RecaleSlop tag!') 
    try:
        img = img + ds.RescaleIntercept
    except:
        print('No RecaleSlop tag!')
    try:
        WL, WW = [get_first_of_dicom_field_as_int(x) for x in [ds.WindowCenter, ds.WindowWidth]]
        img = window(img, WL=WL, WW=WW)
    except:
        warnings.warn('No WindowCenter or WindowWidth tag')
    
    img -= img.min()
    img /= img.max()
#     if len(img.shape) == 2:
#         img = np.stack([img]*3, -1)
    img = (img*255).astype(np.uint8)
    return img


def cv_registration(img_aligned, img_ref, keypoint_detection='ORB', features=5000):
    
    img1 = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2GRAY) 
    img2 = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY) 
    height, width = img2.shape 
    
    if keypoint_detection=='ORB':
    # Create ORB detector with 5000 features. 
        key_detector = cv2.ORB_create(features) 
    if keypoint_detection=='SIFT':
        key_detector = cv2.xfeatures2d.SIFT_create(features) 
    # Find keypoints and descriptors. 
    # The first arg is the image, second arg is the mask 
    #  (which is not reqiured in this case). 
    kp1, d1 = key_detector.detectAndCompute(img1, None) 
    kp2, d2 = key_detector.detectAndCompute(img2, None) 
     
    # Match features between the two images. 
    # We create a Brute Force matcher with  
    # Hamming distance as measurement mode. 
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True) 

    # Match the two sets of descriptors. 
    d1 = d1.astype(np.uint8)
    d2 = d2.astype(np.uint8)
    matches = matcher.match(d1, d2) 
  
    # Sort matches on the basis of their Hamming distance. 
    matches.sort(key = lambda x: x.distance) 

    # Take the top 90 % matches forward. 
    matches = matches[:int(len(matches)*90)] 
    no_of_matches = len(matches) 

    # Define empty matrices of shape no_of_matches * 2. 
    p1 = np.zeros((no_of_matches, 2)) 
    p2 = np.zeros((no_of_matches, 2)) 

    for i in range(len(matches)): 
        p1[i, :] = kp1[matches[i].queryIdx].pt 
        p2[i, :] = kp2[matches[i].trainIdx].pt 

    # Find the homography matrix. 
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC) 

    # Use this matrix to transform the 
    # colored image wrt the reference image. 
    transformed_img = cv2.warpPerspective(img_aligned, 
                        homography, (width, height)) 

    return transformed_img, homography



def get_lesion_map_roi(dcm_file):
    ds = pydicom.read_file(dcm_file)
    assert(ds.SeriesDescription=='Lesion map - AutoMIStar')
    img = ds.pixel_array.copy()
    assert(img.shape==(320,320,3))
    img[:,:25] = 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)[1][:,:,0]
    gray = cv2.inpaint(gray , mask, 1, cv2.INPAINT_NS)
    gray = cv2.blur(gray,(9,9))
    ret, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((7, 7), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=5)
    opening[opening>0] = 1
    
    roi = img*np.stack([(1-opening)]*3,-1)
    return roi, ds.pixel_array
    
def get_ncct_roi(dcm_file):
    img = load_dicom_img(dcm_file)
    assert(img.shape==(512, 512, 3))
    img = img[:,:,0]
    img = cv2.blur(img, (9,9))
    ret, thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((7, 7), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=5)
    opening[opening>0] = 1
    roi = img*np.stack([(1-opening)]*3,-1)
    return roi, ds.pixel_array