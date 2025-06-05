# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 08:56:22 2025

@author: Ryan.Larson

Functions for taking in a pair of structured CSV files, computing point clouds,
and determining the deflection distances between the point clouds
"""

import numpy as np
import pandas as pd
import open3d as o3d
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def direction_from_angle(theta):
    direction = np.array([np.cos(theta), np.sin(theta), 0])
    direction = list(direction / np.linalg.norm(direction))
    return direction

def calculate_point(angle, distance, ref_point):
    x_ref = ref_point[0]
    y_ref = ref_point[1]
    
    x = distance * np.cos(angle) + x_ref
    y = distance * np.sin(angle) + y_ref
    z = ref_point[2]
    
    return np.array([x, y, z])

def calculate_points(df, ref_heights):
    columns = list(df.columns)
    
    points = []
    for col in columns:
        if col == "Angle (deg)":
            continue
        
        for char in col:
            if char.isdigit():
                sensor_num = int(char)
                break
        ref_point = np.array([0, 0, ref_heights[sensor_num]])
        
        for i, row in df.iterrows():
            angle = np.radians(row['Angle (deg)'])
            distance = row[col]
            point = calculate_point(angle, distance, ref_point)
            points.append(point)
            
    return points
    
def get_point_cloud(df, ref_heights):
    points = calculate_points(df, ref_heights)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    return pcd

def measure_displacement(undeformed_pcd, deformed_pcd):
    # Convert point clouds to NumPy arrays
    undeformed_points = np.asarray(undeformed_pcd.points)
    deformed_points = np.asarray(deformed_pcd.points)
    
    # Build a k-d tree for fast nearest-neighbor search
    tree = KDTree(deformed_points)
    
    # Find nearest neighbors in the deformed scan
    distances, indices = tree.query(undeformed_points)
    
    # Compute displacement vectors
    displacements = deformed_points[indices] - undeformed_points
    
    # Compute Euclidean displacement magnitude per point
    displacement_magnitudes = np.linalg.norm(displacements, axis=1)
    
    # Normalize displacement magnitudes for colormap
    norm = mcolors.Normalize(vmin=np.min(displacement_magnitudes), vmax=np.max(displacement_magnitudes))
    colormap = cm.viridis
    colors = colormap(norm(displacement_magnitudes))[:, :3]  # RGB only
    
    # Assign colors to the deformed point cloud
    deformed_pcd.colors = o3d.utility.Vector3dVector(colors)    
    
    # Print summary statistics
    print(f"Max displacement: {np.max(displacement_magnitudes):.2f}")
    print(f"Min displacement: {np.min(displacement_magnitudes):.2f}")
    print(f"Average displacement: {np.mean(displacement_magnitudes):.2f}")
    
    return displacement_magnitudes, deformed_pcd


unloaded_csv = "unloaded.csv"
loaded_csv = "loaded.csv"

df_unloaded = pd.read_csv(unloaded_csv)
df_loaded = pd.read_csv(loaded_csv)

columns = list(df_unloaded.columns)

ref_heights = [6, 18, 30, 42, 54]

pcd_unloaded = get_point_cloud(df_unloaded, ref_heights)
pcd_loaded = get_point_cloud(df_loaded, ref_heights)

displacement_magnitudes, colored_pcd = measure_displacement(pcd_unloaded, pcd_loaded)

o3d.visualization.draw_geometries([colored_pcd],
                                  window_name="Deformed Point Cloud - Colored by Displacement",)

# # Visualize the point cloud
# o3d.visualization.draw_geometries([pcd])