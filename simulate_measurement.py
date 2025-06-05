# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 16:48:58 2025

@author: Ryan.Larson
"""

import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.colors as mcolors


# Function to simulate the measurement at a given angle
def simulate_measurement(ref_heights, theta_range_rad, phi_range_rad, scene, distance_noise=0.0, angle_noise=0.0):    
    directions = []
    
    for theta in theta_range_rad:
        for phi in phi_range_rad:
            # Add noise to angles
            theta_noisy = theta + np.random.uniform(-angle_noise, angle_noise)
            phi_noisy = phi + np.random.uniform(-angle_noise, angle_noise)
            
            # Convert spherical coordinates (theta, phi) to cartesian direction
            direction = np.array([
                np.cos(phi_noisy) * np.sin(theta_noisy),
                np.sin(phi_noisy),
                np.cos(phi_noisy) * np.cos(theta_noisy)
            ])
        
            # Normalize the direction to create the ray
            direction = list(direction / np.linalg.norm(direction))
            # direction = direction / np.linalg.norm(direction)
            
            directions.append(direction)
    
    ray_vals = []
    for ref_height in ref_heights:
        ref_point = [0, ref_height, 0]
        for direction in directions:
            ray_vals.append(ref_point + direction)
    # ray_vals = [ref_point + direction for direction in directions]
    
    rays = o3d.core.Tensor(ray_vals, dtype=o3d.core.Dtype.Float32)
    # ray = o3d.core.Tensor([np.concatenate((ref_point, direction))], dtype=o3d.core.Dtype.Float32)
    
    ans = scene.cast_rays(rays)
    
    hit = ans['t_hit'].isfinite()
    points = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1))
    pcd = o3d.t.geometry.PointCloud(points)
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=ref_point)
    # o3d.visualization.draw_geometries([pcd.to_legacy()],
    #                               front=[0.5, 0.86, 0.125],
    #                               lookat=[0.23, 0.5, 2],
    #                               up=[-0.63, 0.45, -0.63],
    #                               zoom=0.7)
    return pcd

def simulate_offset_measurement(ref_heights, x_offset, y_offset, theta_range_rad, scene):    
    """Assumes X is right, Y is forward, Z is up."""
    directions = []
    
    # for theta in theta_range_rad:
    #     # Convert spherical coordinates (theta, phi) to cartesian direction
    #     direction = np.array([
    #         np.cos(theta),
    #         np.sin(theta),
    #         0,
    #     ])
    
    #     # Normalize the direction to create the ray
    #     direction = list(direction / np.linalg.norm(direction))
    #     # direction = direction / np.linalg.norm(direction)
        
    #     directions.append(direction)
    
    ray_vals = []
    for ref_height in ref_heights:
        for theta in theta_range_rad:
            r = np.sqrt(x_offset**2 + y_offset**2)
            ref_point = [r*np.cos(theta), r*np.sin(theta), ref_height]
            
            direction = np.array([
                np.cos(theta),
                np.sin(theta),
                0])
            
            direction = list(direction / np.linalg.norm(direction))
            
            ray_vals.append(ref_point + direction)
            
        # for direction in directions:
        #     ray_vals.append(ref_point + direction)
    # ray_vals = [ref_point + direction for direction in directions]
    
    rays = o3d.core.Tensor(ray_vals, dtype=o3d.core.Dtype.Float32)
    # ray = o3d.core.Tensor([np.concatenate((ref_point, direction))], dtype=o3d.core.Dtype.Float32)
    
    ans = scene.cast_rays(rays)
    
    hit = ans['t_hit'].isfinite()
    points = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1))
    pcd = o3d.t.geometry.PointCloud(points)
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=ref_point)
    # o3d.visualization.draw_geometries([pcd.to_legacy()],
    #                               front=[0.5, 0.86, 0.125],
    #                               lookat=[0.23, 0.5, 2],
    #                               up=[-0.63, 0.45, -0.63],
    #                               zoom=0.7)
    return pcd

def measure_displacement(undeformed_pcd, deformed_pcd):
    # Convert point clouds to NumPy arrays
    undeformed_points = np.asarray(o3d.t.geometry.PointCloud.to_legacy(undeformed_pcd).points)
    deformed_points = np.asarray(o3d.t.geometry.PointCloud.to_legacy(deformed_pcd).points)
    
    # Build a k-d tree for fast nearest-neighbor search
    tree = KDTree(deformed_points)
    
    # Find nearest neighbors in the deformed scan
    distances, indices = tree.query(undeformed_points)
    
    # Compute displacement vectors
    displacements = deformed_points[indices] - undeformed_points
    
    # Compute Euclidean displacement magnitude per point
    displacement_magnitudes = np.linalg.norm(displacements, axis=1)
    displacement_magnitudes = 1000*displacement_magnitudes
    
    # Normalize displacement magnitudes for colormap
    norm = mcolors.Normalize(vmin=np.min(displacement_magnitudes), vmax=np.max(displacement_magnitudes))
    colormap = cm.viridis
    colors = colormap(norm(displacement_magnitudes))[:, :3]  # Get RGB values (discard alpha)
    
    # Assign colors to the deformed point cloud
    deformed_pcd_legacy = o3d.t.geometry.PointCloud.to_legacy(deformed_pcd)
    deformed_pcd_legacy.colors = o3d.utility.Vector3dVector(colors)    
    
    # Print summary statistics
    print(f"Max displacement: {np.max(displacement_magnitudes)} mm")
    print(f"Min displacement: {np.min(displacement_magnitudes)} mm")
    print(f"Average displacement: {np.mean(displacement_magnitudes)} mm")
    
    return displacement_magnitudes, deformed_pcd_legacy

if __name__ == "__main__":
    # Load your mesh file (change the file path)
    # mesh = o3d.io.read_triangle_mesh("Undeformed.stl")
    # mesh_deformed = o3d.io.read_triangle_mesh("Deformed.stl")
    mesh = o3d.io.read_triangle_mesh("Denali6038.stl")
    mesh_deformed = o3d.io.read_triangle_mesh("Denali6038_deformed.stl")
    
    # # Ensure the mesh is valid
    # if not mesh.is_triangle_mesh():
    #     print("Not a valid triangle mesh")
    
    # Reference point in world coordinates
    # ref_point = [0, -500, 0]  # Example: Origin, you can change this
    # ref_heights = [0.5]
    ref_heights = list(np.linspace(0.0,1.0,10))
    # ref_heights = [0, -125, -250, -375, -500]
    # ref_point = np.array([0, 0, 0])  # Example: Origin, you can change this
    
    # Angular range (in degrees), adjusting as needed
    minaz = 2
    maxaz = 178
    npts = 1000
    print(f'Azimuth Resolution: {(maxaz-minaz)/npts:.2f} deg')
    theta_range = np.linspace(minaz, maxaz, npts)  # Azimuth angles (around the object)
    phi_range = np.asarray([0.0])   # Elevation angles (up/down)
    # phi_range = np.linspace(-90, 90, 200)   # Elevation angles (up/down)
    
    # Convert to radians
    theta_range_rad = np.radians(theta_range)
    phi_range_rad = np.radians(phi_range)
    
    scene = o3d.t.geometry.RaycastingScene()
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    mesh_id = scene.add_triangles(mesh)
    
    pcd = simulate_offset_measurement(ref_heights, 0.2, 0.2, theta_range_rad, scene)
    # pcd = simulate_measurement(ref_heights, theta_range_rad, phi_range_rad, scene, distance_noise=0.0, angle_noise=0.0)
    
    scene = o3d.t.geometry.RaycastingScene()
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_deformed)
    mesh_id = scene.add_triangles(mesh)
    
    pcd_deformed = simulate_offset_measurement(ref_heights, 0.2, 0.2, theta_range_rad, scene)
    # pcd_deformed = simulate_measurement(ref_heights, theta_range_rad, phi_range_rad, scene, distance_noise=0.0, angle_noise=0.0)

    displacement_magnitudes, colored_pcd = measure_displacement(pcd, pcd_deformed)
    
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0,0,0])
    # Visualize the colored point cloud
    o3d.visualization.draw_geometries([colored_pcd],
                                      window_name="Deformed Point Cloud - Colored by Displacement",
                                      front=[0, -1.0, 1.0],
                                      lookat=[0.0, 0.0, 0.5],
                                      up=[0, 0, 1],
                                      zoom=0.7)