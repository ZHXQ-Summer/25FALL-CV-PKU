# This is a raw framework for image stitching using Harris corner detection.
# For libraries you can use modules in numpy, scipy, cv2, os, etc.
import numpy as np
from scipy import ndimage, spatial
import cv2
from os import listdir
import matplotlib.pyplot as plt
import math
import random
import utils
from utils import convolve_2d, gaussian_blur_kernel_2d, pad_with_zeros, cross_correlation_2d
IMGDIR = 'Problem2Images'


def gradient_x(img):
    # convert img to grayscale
    # should we use int type to calclate gradient?
    # should we conduct some pre-processing to remove noise? which kernel should we pply?
    # which kernel should we choose to calculate gradient_x?
    # TODO
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    grad_x = convolve_2d(img, kernel_x)
    return grad_x

def gradient_y(img):
    # TODO
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    grad_y = convolve_2d(img, kernel_y)
    return grad_y

def harris_response(img, alpha, win_size):
    # In this function you are going to claculate harris response R.
    # Please refer to 04_Feature_Detection.pdf page 32 for details. 
    # You have to discover how to calculate det(M) and trace(M), and
    # remember to smooth the gradients. 
    # Avoid using too much "for" loops to speed up.
    # TODO
    if len(img.shape) == 3:
        gray = np.mean(img, axis=2).astype(np.float32)
    else:
        gray = img.astype(np.float32)
    gra_x= gradient_x(gray)
    gra_y= gradient_y(gray)
    gra_xx = gra_x*gra_x
    gra_yy = gra_y*gra_y
    gra_xy = gra_x*gra_y
    H,W=gray.shape
    filter=gaussian_blur_kernel_2d(1, win_size, win_size)
    gra_xx=convolve_2d(gra_xx,filter)
    gra_yy=convolve_2d(gra_yy,filter)
    gra_xy=convolve_2d(gra_xy,filter)
    R = np.zeros((H, W))
    det_M = gra_xx * gra_yy - gra_xy** 2  
    trace_M = gra_xx + gra_yy 
    R = det_M - alpha * (trace_M **2)  
                               
    return R

def corner_selection(R, thresh, min_dist):
    # non-maximal suppression for R to get R_selection and transform selected corners to list of tuples
    # hint: 
    #   use ndimage.maximum_filter()  to achieve non-maximum suppression
    #   set those which aren’t **local maximum** to zero.
    # TODO
    pix=[]
    R_m=ndimage.maximum_filter(R,size=(2*min_dist+1,2*min_dist+1))
    R_s=(R==R_m)*(R>thresh)*R
    for i in range(np.shape(R_s)[0]):
        for j in range(np.shape(R_s)[1]):
            if R_s[i][j]>0:
                pix.append((j,i))
    return pix

def histogram_of_gradients(img, pix):
    # no template for coding, please implement by yourself.
    # You can refer to implementations on Github or other websites
    # Hint: 
    #   1. grad_x & grad_y
    #   2. grad_dir by arctan function
    #   3. for each interest point, choose m*m blocks with each consists of m*m pixels
    #   4. I divide the region into n directions (maybe 8).
    #   5. For each blocks, calculate the number of derivatives in those directions and normalize the Histogram. 
    #   6. After that, select the prominent gradient and take it as principle orientation.
    #   7. Then rotate it’s neighbor to fit principle orientation and calculate the histogram again. 
    # TODO
    if(len(img.shape)==3):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray=gray.astype(np.float32)
    else:
        gray=img.astype(np.float32)
    gra_x= gradient_x(gray)
    gra_y= gradient_y(gray)
    gra_dir=np.arctan2(gra_y,gra_x)*180/math.pi
    gra_len=np.sqrt(gra_x**2+gra_y**2)
    gra_dir+=180
    features=[]
    cell_size=4
    cell_pixel=16
    for (x, y) in pix:
        block_hist=[]
        h, w = gray.shape
        half = cell_pixel // 2
        x1, x2 = max(0, x - half), min(w, x + half)
        y1, y2 = max(0, y - half), min(h, y + half)
        block_len = gra_len[y1:y2, x1:x2]
        block_dir = gra_dir[y1:y2, x1:x2]
        bh, bw = block_len.shape
        
        if bh < cell_pixel or bw < cell_pixel:
            p_h=cell_pixel-bh
            p_w=cell_pixel-bw
            if p_h%2==0:
                p_h=p_h//2
            else:
                p_h=p_h//2+1
            if p_w%2==0:
                p_w=p_w//2
            else:
                p_w=p_w//2+1
            block_len=pad_with_zeros(block_len,p_h,p_w)
            block_dir=pad_with_zeros(block_dir,p_h,p_w)
        bin_w=360/8
        all_hist=np.zeros(8)
        for i in range(0, cell_pixel):
            for j in range(0, cell_pixel):
                bin_idx = int(block_dir[i][j] // bin_w)
                if bin_idx == 8:
                    bin_idx = 7
                all_hist[bin_idx] += block_len[i][j]
        all_hist /= (np.linalg.norm(all_hist) + 1e-10)
        principle_ori=np.argmax(all_hist)*bin_w+bin_w/2
        for i in range(0, cell_pixel, cell_size):
            for j in range(0, cell_pixel, cell_size):
                cell_len = block_len[i:i+cell_size, j:j+cell_size]
                cell_dir = block_dir[i:i+cell_size, j:j+cell_size]
                hist = np.zeros(8)
                for m in range(cell_size):
                    for n in range(cell_size):
                        bin_idx = int((cell_dir[m][n]-principle_ori+360)%360 // bin_w)
                        if bin_idx == 8:
                            bin_idx = 7
                        hist[bin_idx] += cell_len[m][n]
                hist /= (np.linalg.norm(hist) + 1e-10)
                block_hist.append(hist)
        block_hist=np.array(block_hist).flatten()
        block_hist /= (np.linalg.norm(block_hist) + 1e-10)
        features.append(block_hist)
    features=np.array(features)
    return features


def feature_matching(img_1, img_2):
    R1 = harris_response(img_1, 0.04, 5)
    R2 = harris_response(img_2, 0.04, 5)
    cor1 = corner_selection(R1, 0.01*np.max(R1), 5)
    #cor2 = corner_selection(R2, 0.01*np.max(R1), 5)
    cor2 = corner_selection(R2, 0.01*np.max(R1), 5)
    fea1 = histogram_of_gradients(img_1, cor1)
    fea2 = histogram_of_gradients(img_2, cor2)
    dis = spatial.distance.cdist(fea1, fea2, metric='euclidean')
    threshold = 0.92
    print(len(cor1))
    print(len(cor2))
    pixels_1 = []
    pixels_2 = []
    p1, p2 = np.shape(dis)
    if p1 < p2:
        for p in range(p1):
            dis_min = np.min(dis[p])
            pos = np.argmin(dis[p])
            dis[p][pos] = np.max(dis)
            if dis_min/np.min(dis[p]) <= threshold:
                pixels_1.append(cor1[p])
                pixels_2.append(cor2[pos])
                dis[:, pos] = np.max(dis)

    else:
        for p in range(p2):
            dis_min = np.min(dis[:, p])
            pos = np.argmin(dis[:, p])
            dis[pos][p] = np.max(dis)
            if dis_min/np.min(dis[:, p]) <= threshold:
                pixels_2.append(cor2[p])
                pixels_1.append(cor1[pos])
                dis[pos] = np.max(dis)
    min_len = min(np.shape(cor1)[0], np.shape(cor2)[0])
    rate = np.shape(pixels_1)[0]/min_len
    print(np.shape(pixels_1)[0])
    print(rate)
    assert rate >= 0.03, "Fail to Match!"
    return pixels_1, pixels_2

def test_matching():    
    img_1 = cv2.imread(f'{IMGDIR}/4.jpg')
    img_2 = cv2.imread(f'{IMGDIR}/5.jpg')

    img_gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    img_gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    pixels_1, pixels_2 = feature_matching(img_1, img_2)

    H_1, W_1 = img_gray_1.shape
    H_2, W_2 = img_gray_2.shape

    img = np.zeros((max(H_1, H_2), W_1 + W_2, 3))
    img[:H_1, :W_1, (2, 1, 0)] = img_1 / 255
    img[:H_2, W_1:, (2, 1, 0)] = img_2 / 255
    
    plt.figure(figsize=(20, 10), dpi=300)
    plt.imshow(img)

    N = len(pixels_1)
    for i in range(N):
        x1, y1 = pixels_1[i]
        x2, y2 = pixels_2[i]
        plt.plot([x1, x2+W_1], [y1, y2])

    # plt.show()
    plt.savefig('test.jpg')

def compute_homography(pixels_1, pixels_2):
    # compute the best-fit homography using the Singular Value Decomposition (SVD)
    # homography matrix is a (3,3) matrix consisting rotation, translation and projection information.
    # consider how to form matrix A for U, S, V = np.linalg.svd((np.transpose(A)).dot(A))
    # homo_matrix = np.reshape(V[np.argmin(S)], (3, 3))
    # TODO
    A=np.zeros((2*len(pixels_1),9))
    for i in range(len(pixels_1)):
        p_1=pixels_1[i]
        p_2=pixels_2[i]
        x,y=p_1
        xx,yy=p_2
        A[2*i]=np.array([x,y,1,0,0,0,-xx*x,-xx*y,-xx])
        A[2*i+1]=np.array([0,0,0,x,y,1,-yy*x,-yy*y,-yy])
    U, S, VT = np.linalg.svd(A)
    homo_matrix = np.reshape(VT[np.argmin(S)], (3, 3))
    #homo_matrix/=(np.linalg.norm(homo_matrix)+1e-6)
    if(homo_matrix[2][2]<1e-10):
        homo_matrix[2][2]+=1e-6
    homo_matrix/=(homo_matrix[2][2]+1e-6)
    return homo_matrix

def align_pair(pixels_1, pixels_2):
    # utilize \verb|homo_coordinates| for homogeneous pixels
    # and \verb|compute_homography| to calulate homo_matrix
    # implement RANSAC to compute the optimal alignment.
    # you can refer to implementations online.
    best_inlier=[]
    est_homo=None
    max_it=2000
    #min_inliers=10
    min_inliers=10
    #原来是5
    thrs=10.0
    for _ in range(max_it):
        sample=random.sample(range(len(pixels_1)),4)
        sp1=[pixels_1[i] for i in sample]
        sp2=[pixels_2[i] for i in sample]
        homo=compute_homography(sp1,sp2)
        inliers=[]
        for i in range(len(pixels_1)):
            p1=np.array([pixels_1[i][0],pixels_1[i][1],1],dtype=np.float32)
            p2=homo@p1
            np2=(p2/p2[2])[:2]
            err=np.linalg.norm(np2-np.array(pixels_2[i]))
            if(err<=thrs):
                inliers.append(i)
        if(len(inliers)>len(best_inlier) and len(inliers)>min_inliers):
            best_inlier = inliers
            est_homo = compute_homography(
                [pixels_1[i] for i in inliers],
                [pixels_2[i] for i in inliers]
            )
    est_homo=(est_homo.astype(np.float32))
    return est_homo

def stitch_blend(img_1, img_2, est_homo):
    # hint: 
    # First, project four corner pixels with estimated homo-matrix
    # and converting them back to Cartesian coordinates after normalization.
    # Together with four corner pixels of the other image, we can get the size of new image plane.
    # Then, remap both image to new image plane and blend two images using Alpha Blending.
    h1, w1, d1 = np.shape(img_1)  # d=3 RGB
    h2, w2, d2 = np.shape(img_2)
    p1 = est_homo.dot(np.array([0, 0, 1]))
    p2 = est_homo.dot(np.array([0, h1, 1]))
    p3 = est_homo.dot(np.array([w1, 0, 1]))
    p4 = est_homo.dot(np.array([w1, h1, 1]))
    p1 = np.int16(p1/p1[2])
    p2 = np.int16(p2/p2[2])
    p3 = np.int16(p3/p3[2])
    p4 = np.int16(p4/p4[2])
    x_min = min(0, p1[0], p2[0], p3[0], p4[0])
    x_max = max(w2, p1[0], p2[0], p3[0], p4[0])
    y_min = min(0, p1[1], p2[1], p3[1], p4[1])
    y_max = max(h2, p1[1], p2[1], p3[1], p4[1])
    x_range = np.arange(x_min, x_max+1, 1)
    y_range = np.arange(y_min, y_max+1, 1)
    x, y = np.meshgrid(x_range, y_range)
    x = np.float32(x)
    y = np.float32(y)
    homo_inv = np.linalg.pinv(est_homo)
    trans_x = homo_inv[0, 0]*x+homo_inv[0, 1]*y+homo_inv[0, 2]
    trans_y = homo_inv[1, 0]*x+homo_inv[1, 1]*y+homo_inv[1, 2]
    trans_z = homo_inv[2, 0]*x+homo_inv[2, 1]*y+homo_inv[2, 2]
    trans_x = trans_x/trans_z
    trans_y = trans_y/trans_z
    est_img_1 = cv2.remap(img_1, trans_x, trans_y, cv2.INTER_LINEAR)
    est_img_2 = cv2.remap(img_2, x, y, cv2.INTER_LINEAR)
    alpha1 = cv2.remap(np.ones(np.shape(img_1)), trans_x,
                       trans_y, cv2.INTER_LINEAR)
    alpha2 = cv2.remap(np.ones(np.shape(img_2)), x, y, cv2.INTER_LINEAR)
    alpha = alpha1+alpha2
    alpha[alpha == 0] = 2
    alpha1 = alpha1/alpha
    alpha2 = alpha2/alpha
    est_img = est_img_1*alpha1 + est_img_2*alpha2
    return est_img


def generate_panorama(ordered_img_seq):
    len = np.shape(ordered_img_seq)[0]
    mid = int(len/2) # middle anchor
    i = mid-1
    j = mid+1
    principle_img = ordered_img_seq[mid]
    while(j < len):
        pixels1, pixels2 = feature_matching(ordered_img_seq[j], principle_img)
        homo_matrix = align_pair(pixels1, pixels2)
        principle_img = stitch_blend(
            ordered_img_seq[j], principle_img, homo_matrix)
        principle_img=np.uint8(principle_img)
        j = j+1  
    while(i >= 0):
        pixels1, pixels2 = feature_matching(ordered_img_seq[i], principle_img)
        homo_matrix = align_pair(pixels1, pixels2)
        principle_img = stitch_blend(
            ordered_img_seq[i], principle_img, homo_matrix)
        principle_img=np.uint8(principle_img)
        i = i-1  
    est_pano = principle_img
    return est_pano

if __name__ == '__main__':
    # make image list
    # call generate panorama and it should work well
    # save the generated image following the requirements
    # 在main函数中调用测试
    #test_matching()
    
    # an example
    '''img_1 = cv2.imread(f'{IMGDIR}/panoramas/library/1.jpg')
    img_2 = cv2.imread(f'{IMGDIR}/panoramas/library/2.jpg')
    img_3 = cv2.imread(f'{IMGDIR}/panoramas/library/3.jpg')
    img_4 = cv2.imread(f'{IMGDIR}/panoramas/library/4.jpg')
    #img_5 = cv2.imread(f'{IMGDIR}/panoramas/Xue-Mountain-Entrance/DSC_0175.jpg')
    img_list=[]
    img_list.append(img_1)
    img_list.append(img_2)
    img_list.append(img_3)
    img_list.append(img_4)
    #img_list.append(img_5)
    pano = generate_panorama(img_list)'''
    '''for i in range(1,4):
        img00=cv2.imread(f'{IMGDIR}/{i}_1.jpg')
        img01=cv2.imread(f'{IMGDIR}/{i}_2.jpg')
        pixels1, pixels2 = feature_matching(img00, img01)
        homo_matrix = align_pair(pixels1, pixels2)
        principle_img = stitch_blend(
            img00, img01, homo_matrix)
        principle_img=np.uint8(principle_img)
        cv2.imwrite(f"outputs/blend_{i}.jpg", principle_img)'''
    #cv2.imwrite("outputs/panorama_t33.jpg", pano)
