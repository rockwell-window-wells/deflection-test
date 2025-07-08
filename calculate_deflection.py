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

def transform_point_angle_data(df):
    """
    Transforms a DataFrame with 8 pairs of Point/Angle columns and associated metadata
    into a long-form DataFrame with one Point and one Angle column.
    
    Parameters:
    - df (pd.DataFrame): Original DataFrame with columns:
        ['Date', 'Time', 'Sensor', 'Point 1', 'Angle 1', ..., 'Point 8', 'Angle 8']
        
    Returns:
    - pd.DataFrame: Transformed long-form DataFrame with columns:
        ['Date', 'Time', 'Sensor', 'Point', 'Angle']
    """
    # List to hold the transformed rows
    transformed_rows = []

    # Iterate over each row of the original dataframe
    for _, row in df.iterrows():
        for i in range(1, 9):
            point = row[f'Point {i}']
            angle = row[f'Angle {i}']
            transformed_rows.append({
                'Date': row['Date'],
                'Time': row['Time'],
                'Sensor': row['Sensor'],
                'Distance (mm)': point,
                'Angle (deg)': angle
            })

    # Create a new DataFrame from the transformed rows
    new_df = pd.DataFrame(transformed_rows)
    return new_df

def get_points_from_df(df, ref_heights):
    df_new = transform_point_angle_data(df)
    
    columns = list(df_new.columns)
    
    points = []
    
    for i, row in df_new.iterrows():
        sensor_num = row['Sensor']
        ref_point = np.array([0, 0, ref_heights[int(sensor_num)]])
        angle = np.radians(row['Angle (deg)'])
        distance = row['Distance (mm)']
        point = calculate_point(angle, distance, ref_point)
        points.append(point)
        
        
    # for col in columns:
    #     if col == "Angle (deg)":
    #         continue
        
    #     for char in col:
    #         if char.isdigit():
    #             sensor_num = int(char)
    #             break
    #     ref_point = np.array([0, 0, ref_heights[sensor_num]])
        
    #     for i, row in df.iterrows():
    #         angle = np.radians(row['Angle (deg)'])
    #         distance = row[col]
    #         point = calculate_point(angle, distance, ref_point)
    #         points.append(point)
            
    return points
    
    
def get_point_cloud(df, ref_heights):
    # points = calculate_points(df, ref_heights)
    points = get_points_from_df(df, ref_heights)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    return pcd

def apply_colors_improved(pcd, displacement_magnitudes, valid_matches=None):
    if valid_matches is None:
        valid_matches = np.ones(len(displacement_magnitudes), dtype=bool)
    
    # Only use valid displacements for normalization
    valid_displacements = displacement_magnitudes[valid_matches]
    
    if len(valid_displacements) == 0:
        return pcd
    
    # Use percentiles for more robust normalization
    vmin = np.percentile(valid_displacements, 5)  # 5th percentile
    vmax = np.percentile(valid_displacements, 95)  # 95th percentile
    
    # Ensure we have some range
    if vmax - vmin < 1e-6:
        vmax = vmin + 1e-6
    
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colormap = cm.viridis
    
    # Initialize all colors to black
    colors = np.zeros((len(displacement_magnitudes), 3))
    
    # Color only valid points
    colors[valid_matches] = colormap(norm(valid_displacements))[:, :3]
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
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
    
    # # Assign colors to the deformed point cloud
    # deformed_pcd.colors = o3d.utility.Vector3dVector(colors)    
    apply_colors_improved(deformed_pcd, displacement_magnitudes)
    
    # Print summary statistics
    print(f"Max displacement: {np.max(displacement_magnitudes):.2f}")
    print(f"Min displacement: {np.min(displacement_magnitudes):.2f}")
    print(f"Average displacement: {np.mean(displacement_magnitudes):.2f}")
    print(f"Displacement range: {np.min(displacement_magnitudes):.6f} to {np.max(displacement_magnitudes):.6f}")
    print(f"Displacement std: {np.std(displacement_magnitudes):.6f}")
    
    return displacement_magnitudes, deformed_pcd

def measure_displacement_fixed(undeformed_pcd, deformed_pcd):
    undeformed_points = np.asarray(undeformed_pcd.points)
    deformed_points = np.asarray(deformed_pcd.points)
    
    tree = KDTree(deformed_points)
    distances, indices = tree.query(undeformed_points)
    
    displacements = deformed_points[indices] - undeformed_points
    displacement_magnitudes = np.linalg.norm(displacements, axis=1)
    
    # Color the UNDEFORMED point cloud instead
    norm = mcolors.Normalize(vmin=np.min(displacement_magnitudes), vmax=np.max(displacement_magnitudes))
    colormap = cm.viridis
    colors = colormap(norm(displacement_magnitudes))[:, :3]
    
    undeformed_pcd.colors = o3d.utility.Vector3dVector(colors)  # Changed this line
    
    return displacement_magnitudes, undeformed_pcd  # Return undeformed with colors


def visualize_with_colorbar(pcd, displacement_magnitudes, title="Point Cloud with Displacement"):
    """Create a matplotlib 3D plot with colorbar"""
    points = np.asarray(pcd.points)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create scatter plot
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                        c=displacement_magnitudes, cmap='viridis', 
                        s=1, alpha=0.8)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label('Displacement (mm)', rotation=270, labelpad=15)
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(title)
    
    plt.show()

# unloaded_csv = "unloaded.csv"
# loaded_csv = "loaded.csv"
# unloaded_csv = "Point cloud test - unloaded.csv"
# loaded_csv = "Point cloud test - loaded.csv"
unloaded_csv = "point cloud test 2 - initial.csv"
loaded_csv = "point cloud test 2 - deflected.csv"

df_unloaded = pd.read_csv(unloaded_csv)
df_loaded = pd.read_csv(loaded_csv)

columns = list(df_unloaded.columns)

ref_heights = [6, 18, 30, 42, 54]

pcd_unloaded = get_point_cloud(df_unloaded, ref_heights)
pcd_loaded = get_point_cloud(df_loaded, ref_heights)

displacement_magnitudes, colored_pcd = measure_displacement_fixed(pcd_unloaded, pcd_loaded)
# displacement_magnitudes, colored_pcd = measure_displacement(pcd_unloaded, pcd_loaded)

visualize_with_colorbar(colored_pcd, displacement_magnitudes)

o3d.visualization.draw_geometries([colored_pcd],
                                   window_name="Deformed Point Cloud - Colored by Displacement",)

# # Visualize the point cloud
# o3d.visualization.draw_geometries([pcd_unloaded])