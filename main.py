# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 00:42:16 2018

@author: SKT
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as fp

def brightness_contrast(img_path = 'sample.jpg', brightness = 0, contrast = 0):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    orig = img.copy()
    
    #algo to apply contrast and brightness
    img = np.int16(img)
    img = img * (contrast/127 + 1) - contrast + brightness
    img = np.clip(img, 0, 255)
    img = np.uint8(img)
        
    plt.subplot(1,2,1)
    plt.imshow(orig)
        
    plt.subplot(1,2,2)
    plt.imshow(img)
    
def negative(img_path = 'sample.jpg'):
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #algo to neagtive
    neg = 255-img
    
    plt.subplot(1,2,1)
    plt.imshow(img, 'gray')
        
    plt.subplot(1,2,2)
    plt.imshow(neg, 'gray')
    
def log_transform(img_path = 'sample.jpg', c = 3.0):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = np.int16(img)
    
    
    #algo to log transform
    log = c * np.log(img + 1)
    img = np.uint8(img)
    
    
    plt.subplot(1,2,1)
    plt.imshow(img, 'gray')
        
    plt.subplot(1,2,2)
    plt.imshow(log, 'gray')
    
def powerlaw_transform(img_path = 'sample.jpg', c = 3.0, gamma = 3.0):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = np.int16(img)
    
    
    #algo to log transform
    log = c * np.power(img, gamma)
    img = np.uint8(img)
    
    
    plt.subplot(1,2,1)
    plt.imshow(img, 'gray')
        
    plt.subplot(1,2,2)
    plt.imshow(log, 'gray')
        
    
def contrast_stretching(img_path = 'sample.jpg'):    

    def apply(channel):    
        min_range = 0
        max_range = 255
        
        min_i = np.min(channel)
        max_i = np.max(channel)
        
        new =  (channel - min_i) * (((max_range - min_range)/(max_i - min_i)) + min_range)
            
        return np.uint8(new)

    img = cv2.imread(img_path)
    b,g,r = cv2.split(img)
    
    new = cv2.merge((apply(b), apply(g), apply(r)))
    
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        
    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(new, cv2.COLOR_BGR2RGB))
    
def intensity_slicing(img_path = 'sample.jpg', slice_dict = {200 : [70, 80]}):
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    orig = img.copy()
    
    for k,v in slice_dict.items():
        r1 = v[0] 
        r2 = v[1]
        img = np.where(np.logical_and(img>=r1, img<=r2), k, img)
        
    plt.subplot(1,2,1)
    plt.imshow(orig, 'gray')
        
    plt.subplot(1,2,2)
    plt.imshow(img, 'gray')
    
def histogram_equilization(img_path = 'sample.jpg'):
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    
    equ = cv2.equalizeHist(img)
    
    plt.subplot(1,2,1)
    plt.imshow(img, 'gray')
        
    plt.subplot(1,2,2)
    plt.imshow(equ, 'gray')    
    
def box_filter(img_path = 'sample.jpg'):
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    kernel = np.ones((3, 3),np.float32)/9
    dst = cv2.filter2D(img,-1,kernel)
        
    plt.subplot(1,2,1)
    plt.imshow(img, 'gray')
        
    plt.subplot(1,2,2)
    plt.imshow(dst, 'gray')    
    
def median_filter(img_path = 'sample.jpg'):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    dst = cv2.medianBlur(img, 3)
    
    plt.subplot(1,2,1)
    plt.imshow(img, 'gray')
        
    plt.subplot(1,2,2)
    plt.imshow(dst, 'gray')    
    
def min_filter(img_path = 'sample.jpg'):
    
    #dilation
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    kernel = np.ones((3,3))
    dst = cv2.erode(img, kernel)
    
    plt.subplot(1,2,1)
    plt.imshow(img, 'gray')
        
    plt.subplot(1,2,2)
    plt.imshow(dst, 'gray')  
    
def max_filter(img_path = 'sample.jpg'):
    
    #erosion
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    kernel = np.ones((3,3))
    dst = cv2.dilate(img, kernel)
    
    plt.subplot(1,2,1)
    plt.imshow(img, 'gray')
        
    plt.subplot(1,2,2)
    plt.imshow(dst, 'gray')  
    
def laplacian(img_path = 'sample.jpg'):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    dst = cv2.Laplacian(img,cv2.CV_64F)
    
    plt.subplot(1,2,1)
    plt.imshow(img, 'gray')
        
    plt.subplot(1,2,2)
    plt.imshow(dst, 'gray') 
    
def gradient(img_path = 'sample.jpg'):
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    sobelx = cv2.Sobel(img, cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F,0,1,ksize=3)
    
    plt.subplot(2,2,1)
    plt.imshow(img, 'gray')
        
    plt.subplot(2,2,3)
    plt.imshow(sobelx, 'gray') 
    
    plt.subplot(2,2,4)
    plt.imshow(sobely, 'gray') 
    
## Functions to go from image to frequency-image and back   
im2freq = lambda data: fp.rfft(fp.rfft(data, axis=0), axis=1)
freq2im = lambda f: fp.irfft(fp.irfft(f, axis=1), axis=0)

def opening(img_path = 'sample.jpg'):
    
    #erosion followed by dilation
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    kernel = np.ones((3,3))
    dst = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
    plt.subplot(1,2,1)
    plt.imshow(img, 'gray')
        
    plt.subplot(1,2,2)
    plt.imshow(dst, 'gray')
    
def closing(img_path = 'sample.jpg'):
    
    #dilation followed by erosion
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    kernel = np.ones((3,3))
    dst = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    
    plt.subplot(1,2,1)
    plt.imshow(img, 'gray')
        
    plt.subplot(1,2,2)
    plt.imshow(dst, 'gray')
        
def boundry_extraction(img_path = 'sample.jpg'):
        
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    dst = cv2.Canny(img, 100, 200)
    
    plt.subplot(1,2,1)
    plt.imshow(img, 'gray')
        
    plt.subplot(1,2,2)
    plt.imshow(dst, 'gray')
    
def region_filling(img_path = 'coins.jpg'):
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    _,thresh = cv2.threshold(img, 127, 255,0)
    _, contours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.ones(img.shape) * 255
    cv2.drawContours(mask, contours, -1, (0,0,0), -1)
    
    plt.subplot(1,2,1)
    plt.imshow(img, 'gray')
        
    plt.subplot(1,2,2)
    plt.imshow(mask, 'gray')
    
def convex_hull(img_path = 'coins1.png'):
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    _,thresh = cv2.threshold(img, 127, 255,0)
    _, contours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
     
    hulls = []
    
    for contour in contours:
        hulls.append(cv2.convexHull(contour))
    
    mask = np.zeros(img.shape)
    cv2.drawContours(mask, hulls, -1, (255, 255, 255), -1)
        
    plt.subplot(1,2,1)
    plt.imshow(img, 'gray')
        
    plt.subplot(1,2,2)
    plt.imshow(mask, 'gray')
    
def segmentation(img_path = 'sample.jpg', K = 5):
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    Z = img.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)
    
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
   
    plt.imshow(res2)
    plt.show()

    def color_segment():
        img = cv2.imread("59.png")
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2HSV)
        color_range_up=np.array([1, 190, 200])
        color_range_down=np.array([18, 255, 255])
        
        light_orange = (1, 190, 200)
        dark_orange = (18, 255, 255)
        
        mask = cv2.inRange(img_hsv,color_range_up,color_range_down)
        result = cv2.bitwise_and(img_rgb,img_rgb, mask=mask)
        
        plt.imshow(result) 
    
    
    
    
    
    
    
