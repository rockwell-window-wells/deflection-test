# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 12:24:56 2025

@author: Ryan.Larson
"""

import numpy as np
import cv2

def generate_arc_points(arc_list, d):
    """
    Generate points along a set of arcs, spaced by distance d.
    
    Parameters:
        arc_list: List of tuples (xc, yc, r, theta_s, theta_e)
            - xc, yc: Center of the arc
            - r: Radius of the arc
            - theta_s: Start angle in degrees
            - theta_e: End angle in degrees
        d: Desired spacing between points
    
    Returns:
        Numpy array of shape (N, 2) representing the point cloud.
    """
    points = []
    
    for xc, yc, r, theta_s, theta_e in arc_list:
        # Convert angles to radians
        theta_s = np.radians(theta_s)
        theta_e = np.radians(theta_e)
        
        # Ensure correct traversal direction
        if theta_e < theta_s:
            theta_e += 2 * np.pi  # Handle wraparound for full circles
        
        # Compute arc length
        L = r * abs(theta_e - theta_s)
        
        # Number of points to generate
        N = max(1, int(L // d))
        
        # Generate points along the arc
        theta_values = np.linspace(theta_s, theta_e, N + 1)
        for theta in theta_values:
            x = xc + r * np.cos(theta)
            y = yc + r * np.sin(theta)
            points.append([x, y])
    
    return np.array(points)

def compute_emd(hist1, hist2):
    """
    Computes the Earth Mover's Distance (EMD) between two histograms.
    
    Parameters:
        hist1: np.array, histogram counts of dataset 1
        hist2: np.array, histogram counts of dataset 2
        
    Returns:
        EMD value (float)
    """
    # Normalize histograms to probability distributions
    hist1 = hist1.astype(np.float32) / hist1.sum()
    hist2 = hist2.astype(np.float32) / hist2.sum()

    # Convert histograms into cumulative distributions with weights
    bins = np.arange(len(hist1), dtype=np.float32)  # Bin positions

    # Reshape to ensure correct input format (N, 1)
    bins = bins.reshape(-1, 1)
    hist1 = hist1.reshape(-1, 1)
    hist2 = hist2.reshape(-1, 1)

    sig1 = np.hstack((hist1, bins)).astype(np.float32)  # Ensure dtype is float32
    sig2 = np.hstack((hist2, bins)).astype(np.float32)

    # Compute EMD
    emd_value, _, _ = cv2.EMD(sig1, sig2, cv2.DIST_L2)
    return emd_value


# Example usage
normal_arcs = [
    (209.255556809, 61.896405228, 247.132326753, 194.6, 201.7),
    (-7.266374863, -24.526594503, 14.0, 201.7, 262.5),
    (0, 31.003063698, 70.003063698, 262.5, 277.5),
    (7.266374863, -24.526594503, 14.0, 277.5, 338.2),
    (-209.255556809, 61.896405228, 247.132326753,338.2, 345.3)
]

# Skewed arcs
skewed_arcs = [
    (103.436978487, 43.529884316, 140.394177835, 198.3, 211),
    (-4.038883469, -21.066082522, 15, 211, 262.4),
    (3.268346606, 33.449432745, 70.003063698, 262.4, 277.7),
    (10.486999292, 20.420172803, 15.651686282, 277.7, 344.9),
    (-106.85560647, 11.388703853, 137.231130004, 344.9, 355.0)
]

# arc_definitions = [
#     (0, 0, 10, 0, 90),    # Quarter-circle from (10,0) to (0,10)
#     (10, 10, 5, 180, 270)  # Quarter-circle from (5,10) to (10,5)
# ]
distance = 0.05

normal_points = generate_arc_points(normal_arcs, distance)
skewed_points = generate_arc_points(skewed_arcs, distance)


# X Values
bins = [-29.89666211, -26.91245033, -23.92823855, -20.94402677,
       -17.95981498, -14.9756032 , -11.99139142,  -9.00717964,
        -6.02296786,  -3.03875607,  -0.05454429,   2.92966749,
         5.91387927,   8.89809105,  11.88230284,  14.86651462,
        17.8507264 ,  20.83493818,  23.81914996,  26.80336175,
        29.78757353]

normal_histogram, _ = np.histogram(normal_points[:,0], bins=bins)
skewed_histogram, _ = np.histogram(skewed_points[:,0], bins=bins)

emd_dist = compute_emd(normal_histogram, skewed_histogram)

print(f'EMD Value for X shift:\t{emd_dist}')

# Y Values
bins = list(np.arange(-40,20,2))

normal_histogram, _ = np.histogram(normal_points[:,1], bins=bins)
skewed_histogram, _ = np.histogram(skewed_points[:,1], bins=bins)

emd_dist = compute_emd(normal_histogram, skewed_histogram)

print(f'EMD Value for Y shift:\t{emd_dist}')
