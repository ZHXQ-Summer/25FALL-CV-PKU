import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import os
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
def low_pass(img, sigma, size):
    kernel = gaussian_blur_kernel_2d(sigma, size, size)
    return convolve_2d(img, kernel)
def img_subsampling(img,f=2):
    if len(img.shape) == 2:
        return img[::f, ::f]
    elif len(img.shape) == 3:
        return img[::f, ::f, :]
def gaussian_pyramid(image,levels):
    pmid = [image]
    c_image = image
    for _ in range(1, levels):
        blur_image = low_pass(c_image, sigma=1, size=5)
        subsamp_image = img_subsampling(blur_image, f=2)
        pmid.append(subsamp_image)
        c_image = subsamp_image
    return pmid
def main():
    filenames=['frog.jpg','lena.png']
    if not os.path.exists('output'):
        os.makedirs('output')
    for filename in filenames:
        image = np.array(Image.open(filename).convert('RGB'))
        pyramid = gaussian_pyramid(image, levels=4)
        for j in range(1,4):
            Image.fromarray(pyramid[j].astype(np.uint8)).save(os.path.join('output', f'level_{j+1}_{filename}'))
if __name__ == "__main__":
    main()