import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import trimesh
import multiprocessing as mp
from tqdm import tqdm
from typing import Tuple
import math
import time
def normalize_disparity_map(disparity_map):
    '''Normalize disparity map for visualization 
    disparity should be larger than zero
    '''
    return np.maximum(disparity_map, 0.0) / (disparity_map.max() + 1e-10)


def visualize_disparity_map(disparity_map, gt_map, save_path=None):
    '''Visualize or save disparity map and compare with ground truth
    '''
    # Normalize disparity maps
    disparity_map = normalize_disparity_map(disparity_map)
    gt_map = normalize_disparity_map(gt_map)
    # Visualize or save to file
    if save_path is None:
        concat_map = np.concatenate([disparity_map, gt_map], axis=1)
        plt.imshow(concat_map, 'gray')
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        concat_map = np.concatenate([disparity_map, gt_map], axis=1)
        plt.imsave(save_path, concat_map, cmap='gray')


def task1_compute_disparity_map_simple(
    ref_img: np.ndarray,        # shape (H, W)
    sec_img: np.ndarray,        # shape (H, W)
    window_size: int, 
    disparity_range: Tuple[int, int],   # (min_disparity, max_disparity)
    matching_function: str      # can be 'SSD', 'SAD', 'normalized_correlation'
):
    '''Assume image planes are parallel to each other
    Compute disparity map using simple stereo system following the steps:
    1. For each row, scan all pixels in that row
    2. Generate a window for each pixel in ref_img
    3. Search for a disparity (d) within (min_disparity, max_disparity) in sec_img 
    4. Select the best disparity that minimize window difference between ref_img[row, col] and sec_img[row, col - d]
    '''
    start_time = time.time()
    disparity_map = np.zeros_like(ref_img,dtype=np.float32)
    hw=window_size//2
    height, width = ref_img.shape
    total_pixels = (height - 2*hw) * (width - 2*hw)

    with tqdm(total=total_pixels, desc=f"{matching_function}", unit="pixel") as pbar:
        for i in range(hw,ref_img.shape[0]-hw):
            for j in range(hw,ref_img.shape[1]-hw):#遍历左图的每个像素
                best_dis=0
                best_cost=math.inf
                window_ref=ref_img[i-hw:i+hw+1,j-hw:j+hw+1]
                for d in range(disparity_range[0],disparity_range[1]+1):
                    t=j-d
                    if t<hw or t>=ref_img.shape[1]-hw:
                        continue
                    window_sec=sec_img[i-hw:i+hw+1,t-hw:t+hw+1]
                    if matching_function=="SSD":
                        cost=np.sum((window_ref-window_sec)**2)
                    elif matching_function=="SAD":
                        cost=np.sum(np.abs(window_ref-window_sec))
                    else:
                        mean_ref=np.mean(window_ref)
                        mean_sec=np.mean(window_sec)
                        cen_ref=window_ref-mean_ref
                        cen_sec=window_sec-mean_sec
                        num=np.sum(cen_sec*cen_ref)
                        denum=np.sqrt(np.sum(cen_ref**2)*np.sum(cen_sec**2))
                        if(denum<=1e-6):
                            cost=math.inf
                        else:
                            cost=num/denum
                            cost=-cost
                    if cost<best_cost:
                        best_cost=cost
                        best_dis=d
                disparity_map[i,j]=best_dis
                pbar.update(1)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"运行时间: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
    return disparity_map

def task1_simple_disparity(ref_img, sec_img, gt_map, img_name='tsukuba'):
    '''Compute disparity maps for different settings
    '''
    window_sizes = [21]  # Try different window sizes
    disparity_range = (0, 64)  # Determine appropriate disparity range
    matching_functions = ['SSD']  # Try different matching functions
    
    disparity_maps = []
    
    # Generate disparity maps for different settings
    for window_size in window_sizes:
        for matching_function in matching_functions:
            print(f"Computing disparity map for window_size={window_size}, disparity_range={disparity_range}, matching_function={matching_function}")
            disparity_map = task1_compute_disparity_map_simple(
                ref_img, sec_img, 
                window_size, disparity_range, matching_function)
            disparity_maps.append((disparity_map, window_size, matching_function, disparity_range))
            dmin, dmax = disparity_range
            visualize_disparity_map(
                disparity_map, gt_map, 
                save_path=f"output/task1_{img_name}_{window_size}_{dmin}_{dmax}_{matching_function}.png")
    return disparity_maps


def task2_compute_depth_map(disparity_map, baseline, focal_length):
    '''Compute depth map by z = fB / (x - x')
    Note that a disparity less or equal to zero should be ignored (set to zero) 
    '''
    depth_map = np.zeros_like(disparity_map, dtype=np.float32)
    valid_mask = disparity_map > 0
    depth_map[valid_mask] = (baseline * focal_length) / disparity_map[valid_mask]
    return depth_map

def task2_visualize_pointcloud(
    ref_img: np.ndarray,        # shape (H, W, 3) 
    disparity_map: np.ndarray,  # shape (H, W)
    save_path: str = 'output/task2_tsukuba.ply'
):
    '''Visualize 3D pointcloud from disparity map following the steps:
    1. Calculate depth map from disparity
    2. Set pointcloud's XY as image's XY and and pointcloud's Z as depth
    3. Set pointcloud's color as ref_img's color
    4. Save pointcloud to ply files for visualizationh. We recommend to open ply file with MeshLab
    5. Adjust the baseline and focal_length for better performance
    6. You may need to cut some outliers for better performance
    '''
    baseline = 100 * ((np.mean(disparity_map) / 86) ** 1.25)
    focal_length = 100
    threshold = np.mean(disparity_map) - np.std(disparity_map) * 0.5
    depth_map = task2_compute_depth_map(disparity_map, baseline, focal_length)
    rows, cols = ref_img.shape[:2]
    y_coords, x_coords = np.mgrid[0:rows, 0:cols]
    valid_mask = (
        (disparity_map > threshold) &
        (depth_map > 0) &
        (depth_map < 1000)  
    )
    x = x_coords[valid_mask]
    y = -y_coords[valid_mask]  
    z = depth_map[valid_mask]
    colors = ref_img[valid_mask]
    points = np.column_stack([x, y, z])
    pointcloud = trimesh.PointCloud(points, colors)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pointcloud.export(save_path, file_type='ply')


def task3_compute_disparity_map_dp(ref_img, sec_img):
    ''' Conduct stereo matching with dynamic programming
    '''
    start=time.time()
    ref_image = ref_img / 255
    sec_image = sec_img / 255
    disparity_map = np.zeros_like(ref_image, dtype=np.float32)
    rows, cols = ref_image.shape
    max_disparity = 64
    window = 5
    Occlusion = 0.8

    for i in tqdm(range(window, rows - window), desc="DP匹配进度", unit="行"):
        disparity_space_image = np.full((cols, cols), 1e10)
        for left in range(window, cols - window):
            left_window = ref_image[i - window:i + window + 1, left - window:left + window + 1]
            for right in range(max(left - max_disparity, window), left + 1):
                right_window = sec_image[i - window:i + window + 1, right - window: right + window + 1]
                disparity_space_image[left][right] = np.sum((right_window - left_window) ** 2)
        path = np.zeros_like(disparity_space_image)
        c = np.full_like(disparity_space_image, 1e10)
        for j in range(cols):
            c[j][0] = c[0][j] = j * Occlusion
        for x in range(1, cols):
            for y in range(1, cols):
                min1 = c[x - 1][y - 1] + disparity_space_image[x][y]
                min2 = c[x][y - 1] + Occlusion
                min3 = c[x - 1][y] + Occlusion
                c[x][y] = min(min1, min2, min3)
                if c[x][y] == min1:
                    path[x][y] = 1
                elif c[x][y] == min2:
                    path[x][y] = 2
                else:
                    path[x][y] = 3
        left = cols - 1
        right = np.argmin(c[cols - 1])
        while left > 0:
            choice = path[left][right]
            if choice == 1:
                disparity_map[i][left] = left - right
                left -= 1
                right -= 1
            elif choice == 2:
                disparity_map[i][left] = left - right
                right -= 1
            elif choice == 3:
                disparity_map[i][left] = 0
                left -= 1
            else:
                break 
        # occlusion filling
        for j in range(1, cols):
            if disparity_map[i][j] == 0:
                disparity_map[i][j] = disparity_map[i][j - 1]
    elapsed_time = time.time() - start
    
    print(f"\n✅ DP匹配完成!")
    print(f"⏱️  总运行时间: {elapsed_time:.2f} 秒", end="")
    return disparity_map

def main(tasks): 
    
    # Read images and ground truth disparity maps
    moebius_img1 = cv2.imread("data/moebius1.png")
    moebius_img1_gray = cv2.cvtColor(moebius_img1, cv2.COLOR_BGR2GRAY)
    moebius_img2 = cv2.imread("data/moebius2.png")
    moebius_img2_gray = cv2.cvtColor(moebius_img2, cv2.COLOR_BGR2GRAY)
    moebius_gt = cv2.imread("data/moebius_gt.png", cv2.IMREAD_GRAYSCALE)

    tsukuba_img1 = cv2.imread("data/tsukuba1.jpg")
    tsukuba_img1_gray = cv2.cvtColor(tsukuba_img1, cv2.COLOR_BGR2GRAY)
    tsukuba_img2 = cv2.imread("data/tsukuba2.jpg")
    tsukuba_img2_gray = cv2.cvtColor(tsukuba_img2, cv2.COLOR_BGR2GRAY)
    tsukuba_gt = cv2.imread("data/tsukuba_gt.jpg", cv2.IMREAD_GRAYSCALE)
    
    # Task 0: Visualize cv2 Results
    if '0' in tasks:   
        # Compute disparity maps using cv2
        stereo = cv2.StereoBM.create(numDisparities=64, blockSize=15)
        moebius_disparity_cv2 = stereo.compute(moebius_img1_gray, moebius_img2_gray)
        visualize_disparity_map(moebius_disparity_cv2, moebius_gt)
        tsukuba_disparity_cv2 = stereo.compute(tsukuba_img1_gray, tsukuba_img2_gray)
        visualize_disparity_map(tsukuba_disparity_cv2, tsukuba_gt)
        
        if '2' in tasks:
            print('Running task2 with cv2 results ...')
            task2_visualize_pointcloud(tsukuba_img1, tsukuba_disparity_cv2, save_path='output/task2_tsukuba_cv2.ply')

    ######################################################################
    # Note. Running on moebius may take a long time with your own code   #
    # In this homework, you are allowed only to deal with tsukuba images #
    ######################################################################

    # Task 1: Simple Disparity Algorithm
    if '1' in tasks:
        print('Running task1 ...')
        disparity_maps = task1_simple_disparity(tsukuba_img1_gray, tsukuba_img2_gray, tsukuba_gt, img_name='tsukuba')
        
        #####################################################
        # If you want to run on moebius images,             #
        # parallelizing with multiprocessing is recommended #
        #####################################################
        # task1_simple_disparity(moebius_img1_gray, moebius_img2_gray, moebius_gt, img_name='moebius')
        
        if '2' in tasks:
            print('Running task2 with disparity maps from task1 ...')
            for (disparity_map, window_size, matching_function, disparity_range) in disparity_maps:
                dmin, dmax = disparity_range
                task2_visualize_pointcloud(
                    tsukuba_img1, disparity_map, 
                    save_path=f'output/task2_tsukuba_{window_size}_{dmin}_{dmax}_{matching_function}.ply')      
        
    # Task 3: Non-local constraints
    if '3' in tasks:
        print('----------------- Task 3 -----------------')
        tsukuba_disparity_dp = task3_compute_disparity_map_dp(tsukuba_img1_gray, tsukuba_img2_gray)
        visualize_disparity_map(tsukuba_disparity_dp, tsukuba_gt, save_path='output/task3_tsukuba.png')
        
        if '2' in tasks:
            print('Running task2 with disparity maps from task3 ...')
            task2_visualize_pointcloud(tsukuba_img1, tsukuba_disparity_dp, save_path='output/task2_tsukuba_dp.ply')

if __name__ == '__main__':
    # Set tasks to run
    parser = argparse.ArgumentParser(description='Homework 4')
    parser.add_argument('--tasks', type=str, default='0123')
    args = parser.parse_args()

    main(args.tasks)
