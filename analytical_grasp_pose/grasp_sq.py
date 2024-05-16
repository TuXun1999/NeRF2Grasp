import cv2
import numpy as np

import copy
import open3d as o3d
import json

from superquadrics import *
from mesh_process import *
from image_process import *
import os
from scipy.spatial.transform import Rotation as R


'''
Main Program of the whole grasp pose prediction module
'''

# TODO: Add argument parser here
if __name__ == "__main__":
    nerf_scale = 64

    ## The image used to specify the selected point
    img_dir = "../data/nerf/chair_sim_depth"
    img_file = "/images/chair_2_20.png"

    ## Obtain the ray direction of the selected point in space 
    ray_dir, camera_pose, camera_intrinsic = point_select_from_image(img_dir, img_file, nerf_scale=nerf_scale)
    ray_dir = ray_dir / np.linalg.norm(ray_dir)

    ## Create files as input to other modules
    # Specify the ply file
    filename="chair_upper.obj"
    
    # Read the file as a triangular mesh
    mesh = o3d.io.read_triangle_mesh(filename)

    # Part I: create the normalized mesh as the input to superquadrics (in the format of shapenetv2)
    normalize_obj = "model_normalized.obj"
    if not os.path.isfile(normalize_obj): # Correct the coordinate convention if the normalized model is not generated
        # Restore the correct scale & fix up the coordinate issue 
        mesh = coordinate_correction(mesh, filename, nerf_scale)

        # Output the normalized triangular mesh & Obtain the stats
        stats = model_normalized(filename, normalize_obj, \
                                 normalize_stats_file="normalize.npy", stats = None)
    else:
        # If the normalize model already exists, read the parameters directly
        stats = read_normalize_stats("normalize.npy").item()
    
    # TODO: transfer the codes here to generate the predicted superquadrics file



    ## Determine the location of the selected point in space
    pos, point_select_distance = point_select_in_space(camera_pose, ray_dir, mesh)
    print(point_select_distance)
    print(dir)
    print('Visualizing...')

    # TODO: Crop a local region at the specified point





    # Add a sphere to the selected point
    ball_select =  o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    ball_select.scale(1/64 * nerf_scale, [0, 0, 0])

    ball_select.translate((pos[0], pos[1], pos[2]))
    ball_select.paint_uniform_color((1, 0, 0))

    # Plot out the camera frame
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    camera_frame.scale(20/64 * nerf_scale, [0, 0, 0])
    camera_frame.transform(camera_pose)
    

    ## Read the parameters of the superquadrics and import the point

    # Extract out the necessary attributes
    diag = stats['max'] - stats['min']
    norm = np.linalg.norm(diag)
    c = stats['centroid']
    
    # Read the attributes of the predicted sq
    sq_dir = "./test_tmp"
    sq_vertices_original, sq_transformation = read_sq_directory(sq_dir, norm, c)

    
    # Convert sq_verticies_original into a numpy array
    sq_vertices = np.array(sq_vertices_original).reshape(-1, 3)
    
    # Displacement observed from experiments...
    displacement = -np.mean(sq_vertices, axis=0) + np.mean(np.asarray(mesh.vertices), axis=0)
    

    ## Find the sq associated to the selected point
    # Evaluate the transformation of each sq & find the closest sq
    sq_closest, idx = find_sq_closest(pos, sq_transformation, norm, displacement)

    # Delete the point cloud of the associated sq (to draw a new one; avoid point overlapping)
    sq_vertices_original.pop(idx)
    
    # Construct a point cloud representing the reconstructed object mesh
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(sq_vertices_original).reshape(-1, 3) + displacement)
    # Visualize the super-ellipsoids
    pcd.paint_uniform_color((0.0, 0.4, 0.0))

    # Color the associated sq in blue and Complete the whole reconstructed model
    pcd_associated = o3d.geometry.PointCloud()
    pcd_associated.points = o3d.utility.Vector3dVector(sq_closest["points"])
    pcd_associated.paint_uniform_color((0, 0, 1))
    
    # Plot out the fundamental frame of the associated sq
    sq_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    sq_frame.scale(20/64 * nerf_scale, [0, 0, 0])
    sq_frame.transform(sq_closest["transformation"])
    print("Final test")
    print(sq_closest["sq_parameters"])
    print(norm)


    ## Determine the grasp candidates on this sq and visualize them
    # Define the gripper
    scale = 10
    gripper_width = scale
    gripper_length = scale

    # Key points on the gripper
    num_sample = 10
    arm_end = np.array([gripper_length, 0, 0])
    center = np.array([0, 0, 0])
    elbow1 = np.array([0, 0, gripper_width/2])
    elbow2 = np.array([0, 0, -gripper_width/2])
    tip1 = np.array([-gripper_length, 0, gripper_width/2])
    tip2 = np.array([-gripper_length, 0, -gripper_width/2])

    # Construct the gripper
    gripper_points = np.array([
        center,
        arm_end,
        elbow1,
        elbow2,
        tip1,
        tip2
    ])
    gripper_lines = [
        [1, 0],
        [2, 3],
        [2, 4],
        [3, 5]
    ]

    # Sample several points on the gripper
    
    gripper_part1 = np.linspace(arm_end, center, num_sample)
    gripper_part2 = np.linspace(elbow1, tip1, num_sample)
    gripper_part3 = np.linspace(elbow2, tip2, num_sample)
    gripper_part4 = np.linspace(elbow1, elbow2, 2*num_sample)
    gripper_points_sample = np.vstack((gripper_part1, gripper_part2, gripper_part3, gripper_part4))
    print(gripper_points_sample.shape)
    # Pass the gripper attributes to predict the grasp poses
    gripper_attr = {"Type": "parallel", \
                    "Width": gripper_width, "Length": gripper_length, "Thickness": 0.4 * gripper_length,\
                        "Scale": scale}
    grasp_poses = grasp_pose_predict_sq_closest(sq_closest, gripper_attr, norm)
    


    print("Evaluating Grasp Qualities....")
    grasp_cands = [] # All the grasp candidates
    bbox_cands = [] # 
    # Construct the grasp poses at the specified locations,
    # and add them to the visualizer
    for grasp_pose in grasp_poses:
        # Find the grasp pose in the world frame (converted from sq local frame)
        grasp_pose = np.matmul(sq_closest["transformation"], grasp_pose)
       
        
        # Transform the associated points for visualization or collision testing to the correct location
        gripper_points_vis = np.vstack((gripper_points.T, np.ones((1, gripper_points.shape[0]))))
        gripper_points_vis = np.matmul(grasp_pose, gripper_points_vis)
        gripper_points_vis_sample = np.vstack(\
            (gripper_points_sample.T, np.ones((1, gripper_points_sample.shape[0]))))
        gripper_points_vis_sample = np.matmul(grasp_pose, gripper_points_vis_sample)
        
        grasp_pose_lineset = o3d.geometry.LineSet()
        grasp_pose_lineset.points = o3d.utility.Vector3dVector(gripper_points_vis[:-1].T)
        grasp_pose_lineset.lines = o3d.utility.Vector2iVector(gripper_lines)
        
        antipodal_res, bbox = antipodal_test(mesh, grasp_pose, gripper_attr, 5, np.pi/18)
        # Collision Test
        if collision_test_local(mesh, gripper_points_vis_sample[:-1].T, \
                                grasp_pose, gripper_attr, threshold=0.01 * gripper_width):
            grasp_pose_lineset.paint_uniform_color((1, 0, 0))
        else:
            if antipodal_res == True:
                bbox_cands.append(bbox)
                grasp_pose_lineset.paint_uniform_color((0, 1, 0))
            else:
                grasp_pose_lineset.paint_uniform_color((1, 1, 0))
        grasp_cands.append(grasp_pose_lineset)
        
    ## Postlogue
    # Plot out the fundamental frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    frame.scale(20/64 * nerf_scale, [0, 0, 0])


    # Create the window to display everything
    vis= o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    vis.add_geometry(pcd)
    vis.add_geometry(pcd_associated) 
    vis.add_geometry(sq_frame)
    vis.add_geometry(frame)
    vis.add_geometry(camera_frame)
    vis.add_geometry(ball_select)
    for grasp_cand in grasp_cands:
        vis.add_geometry(grasp_cand)
    for bbox_cand in bbox_cands:
        vis.add_geometry(bbox_cand)
    vis.run()
    vis.destroy_window()


