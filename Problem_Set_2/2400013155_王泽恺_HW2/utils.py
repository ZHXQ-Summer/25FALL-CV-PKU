import numpy as np
from scipy import ndimage, spatial
import cv2
from os import listdir
import matplotlib.pyplot as plt
import math

def gaussian_blur_kernel_2d(sigma, h, w):
    kernel = np.zeros((h, w))
    center_h, center_w = h // 2, w // 2
    for i in range(h):
        for j in range(w):
            diff_h = i - center_h
            diff_w = j - center_w
            kernel[i, j] = (1 / (2 * math.pi * sigma ** 2)) * math.exp(-(diff_h ** 2 + diff_w ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    return kernel
def pad_with_zeros(img, pad_height, pad_width):
    if len(img.shape) == 2:
        H, W = img.shape
        p_image = np.zeros((H + 2 * pad_height, W + 2 * pad_width))
        p_image[pad_height:pad_height + H, pad_width:pad_width + W] = img
    elif len(img.shape) == 3:
        H, W, C = img.shape
        p_image = np.zeros((H + 2 * pad_height, W + 2 * pad_width, C))
        p_image[pad_height:pad_height + H, pad_width:pad_width + W, :] = img
    return p_image
def cross_correlation_2d(image,kernel):
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    color_ful=len(image.shape)==3
    if(color_ful):
        H,W,C=image.shape
        re=np.zeros((H,W,C))
        for c in range(C):
            re[:,:,c]=cross_correlation_2d(image[:,:,c],kernel)
        return re
    else:
        H,W=image.shape
        padded_image = pad_with_zeros(image, pad_h,  pad_w)
        re = np.zeros((H, W))
        for i in range(H):
            for j in range(W):
                region = padded_image[i:i + k_h, j:j + k_w]
                re[i, j] = np.sum(region * kernel)
        if image.dtype == np.uint8:
            re = np.clip(re, 0, 255).astype(np.uint8)
        return re
def convolve_2d(img, kernel):
    flipped_kernel = np.flipud(np.fliplr(kernel))
    return cross_correlation_2d(img, flipped_kernel)