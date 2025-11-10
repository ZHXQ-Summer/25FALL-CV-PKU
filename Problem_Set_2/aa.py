# This is a raw framework for image stitching using Harris corner detection.
# For libraries you can use modules in numpy, scipy, cv2, os, etc.
import numpy as np
from scipy import ndimage, spatial
import cv2
from os import listdir
import matplotlib.pyplot as plt


IMGDIR = 'Problem2Images'


def gradient_x(img):
    # 转为灰度图
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    grad_x = ndimage.sobel(img_gray, axis=1, mode='reflect')
    return grad_x

def gradient_y(img):
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    # 使用Sobel算子计算Y轴梯度，边缘采用reflect模式
    grad_y = ndimage.sobel(img_gray, axis=0, mode='reflect')
    return grad_y

def harris_response(img, alpha, win_size):
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    # 计算梯度
    grad_x = gradient_x(img_gray)
    grad_y = gradient_y(img_gray)
    # 计算梯度的平方和乘积
    Ixx = grad_x ** 2
    Iyy = grad_y ** 2
    Ixy = grad_x * grad_y
    # 高斯滤波平滑
    sigma = win_size / 6  # 高斯窗口 sigma 经验值
    Ixx = ndimage.gaussian_filter(Ixx, sigma=sigma, mode='reflect')
    Iyy = ndimage.gaussian_filter(Iyy, sigma=sigma, mode='reflect')
    Ixy = ndimage.gaussian_filter(Ixy, sigma=sigma, mode='reflect')
    # 计算哈里斯响应值R
    det_M = Ixx * Iyy - Ixy ** 2
    tr_M = Ixx + Iyy
    R = det_M - alpha * (tr_M ** 2)
    return R

def corner_selection(R, thresh, min_dist):
    # Apply non-maximum suppression
    R_max = ndimage.maximum_filter(R, size=min_dist)
    
    # Create a mask for local maxima above threshold
    mask = (R == R_max) & (R > thresh)
    
    # Get coordinates of selected corners
    y_coords, x_coords = np.where(mask)
    
    # Create list of tuples (x, y)
    pixels = [(x, y) for x, y in zip(x_coords, y_coords)]
    
    return pixels

def histogram_of_gradients(img, pix):
    # Parameters
    num_bins = 8
    cell_size = 4
    block_size = 2  # 2x2 cells per block
    eps = 1e-7
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        img_gray = img.astype(np.float32)
    
    # Calculate gradients
    grad_x = gradient_x(img_gray)
    grad_y = gradient_y(img_gray)
    
    # Calculate gradient magnitude and orientation
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    orientation = np.arctan2(grad_y, grad_x) * 180 / np.pi
    orientation = np.mod(orientation, 360)  # Convert to 0-360 range
    
    features = []
    
    for x, y in pix:
        # Extract local patch around the keypoint
        half_size = cell_size * block_size // 2
        y_start = max(0, y - half_size)
        y_end = min(img_gray.shape[0], y + half_size)
        x_start = max(0, x - half_size)
        x_end = min(img_gray.shape[1], x + half_size)
        
        # Initialize feature vector for this keypoint
        feature_vector = []
        
        # Process each cell in the block
        for cell_y in range(block_size):
            for cell_x in range(block_size):
                # Calculate cell boundaries
                cell_y_start = y_start + cell_y * cell_size
                cell_y_end = min(cell_y_start + cell_size, y_end)
                cell_x_start = x_start + cell_x * cell_size
                cell_x_end = min(cell_x_start + cell_size, x_end)
                
                # Skip if cell is out of bounds
                if (cell_y_end <= cell_y_start) or (cell_x_end <= cell_x_start):
                    feature_vector.extend([0] * num_bins)
                    continue
                
                # Extract cell region
                cell_mag = magnitude[cell_y_start:cell_y_end, cell_x_start:cell_x_end]
                cell_ori = orientation[cell_y_start:cell_y_end, cell_x_start:cell_x_end]
                
                # Calculate histogram for this cell
                hist, _ = np.histogram(cell_ori, bins=num_bins, range=(0, 360), weights=cell_mag)
                
                # Normalize the histogram
                hist_norm = hist / (np.sqrt(np.sum(hist**2) + eps))
                
                feature_vector.extend(hist_norm)
        
        features.append(feature_vector)
    
    return np.array(features)

def feature_matching(img_1, img_2):
    R1 = harris_response(img_1, 0.04, 9)
    R2 = harris_response(img_2, 0.04, 9)
    cor1 = corner_selection(R1, 0.01*np.max(R1), 5)
    cor2 = corner_selection(R2, 0.01*np.max(R1), 5)
    fea1 = histogram_of_gradients(img_1, cor1)
    fea2 = histogram_of_gradients(img_2, cor2)
    dis = spatial.distance.cdist(fea1, fea2, metric='euclidean')
    threshold = 0.6
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
    assert rate >= 0.03, "Fail to Match!"
    return pixels_1, pixels_2

def test_matching():    
    img_1 = cv2.imread(f'{IMGDIR}/1_1.jpg')
    img_2 = cv2.imread(f'{IMGDIR}/1_2.jpg')

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
    # Convert to homogeneous coordinates
    pts1 = np.array([(p[0], p[1]) for p in pixels_1])
    pts2 = np.array([(p[0], p[1]) for p in pixels_2])
    
    # Normalize points for numerical stability
    def normalize_points(pts):
        mean = np.mean(pts, axis=0)
        std = np.std(pts, axis=0)
        scale = np.sqrt(2) / std
        T = np.array([[scale[0], 0, -scale[0]*mean[0]],
                      [0, scale[1], -scale[1]*mean[1]],
                      [0, 0, 1]])
        pts_homo = np.column_stack((pts, np.ones(len(pts))))
        pts_norm = (T @ pts_homo.T).T
        return pts_norm[:, :2], T
    
    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)
    
    # Build matrix A for SVD
    A = []
    for i in range(len(pts1_norm)):
        x1, y1 = pts1_norm[i]
        x2, y2 = pts2_norm[i]
        A.append([-x1, -y1, -1, 0, 0, 0, x2*x1, x2*y1, x2])
        A.append([0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2])
    
    A = np.array(A)
    
    # Solve using SVD
    U, S, Vt = np.linalg.svd(A)
    H_norm = Vt[-1].reshape(3, 3)
    
    # Denormalize
    H = np.linalg.inv(T2) @ H_norm @ T1
    H = H / H[2, 2]  # Normalize
    
    return H
def align_pair(pixels_1, pixels_2, max_iterations=1000, threshold=5.0):
    best_homo = None
    max_inliers = 0
    
    pts1 = np.array([(p[0], p[1]) for p in pixels_1])
    pts2 = np.array([(p[0], p[1]) for p in pixels_2])
    
    for _ in range(max_iterations):
        # Randomly select 4 point pairs
        indices = random.sample(range(len(pts1)), 4)
        sample_pts1 = [pixels_1[i] for i in indices]
        sample_pts2 = [pixels_2[i] for i in indices]
        
        # Compute homography for this sample
        try:
            H = compute_homography(sample_pts1, sample_pts2)
            
            # Count inliers
            inliers = 0
            for i in range(len(pts1)):
                # Transform point from img1 to img2
                pt1_homo = np.array([pts1[i][0], pts1[i][1], 1])
                pt2_proj = H @ pt1_homo
                pt2_proj = pt2_proj / pt2_proj[2]
                
                # Calculate distance to actual point in img2
                dist = np.sqrt((pt2_proj[0] - pts2[i][0])**2 + (pt2_proj[1] - pts2[i][1])**2)
                
                if dist < threshold:
                    inliers += 1
            
            # Update best homography
            if inliers > max_inliers:
                max_inliers = inliers
                best_homo = H
                
        except np.linalg.LinAlgError:
            continue
    
    # If RANSAC failed, use all points
    if best_homo is None:
        best_homo = compute_homography(pixels_1, pixels_2)
    
    return best_homo


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
    test_matching()
    
    # an example
    img_1 = cv2.imread(f'{IMGDIR}/panoramas/parrington/prtn00.jpg')
    img_2 = cv2.imread(f'{IMGDIR}/panoramas/parrington/prtn01.jpg')
    img_3 = cv2.imread(f'{IMGDIR}/panoramas/parrington/prtn02.jpg')
    img_4 = cv2.imread(f'{IMGDIR}/panoramas/parrington/prtn03.jpg')
    img_5 = cv2.imread(f'{IMGDIR}/panoramas/parrington/prtn04.jpg')
    img_list=[]
    img_list.append(img_1)
    img_list.append(img_2)
    img_list.append(img_3)
    img_list.append(img_4)
    img_list.append(img_5)
    pano = generate_panorama(img_list)
    cv2.imwrite("outputs/panorama_3.jpg", pano)
