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
Main Program of the baseline evaluation Module (Contact GraspNet)
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


    # Create the npy file as the input to Contact GraspNet

    npy_data = {}
    # In order that Contact GraspNet can work normally, we have to scale the model down
    npy_data['xyz_color'] = np.asarray(mesh.vertex_colors)

    mesh_v = np.asarray(mesh.vertices) / nerf_scale
    mesh_v = np.vstack((mesh_v.T, np.ones((1, mesh_v.shape[0]))))
    camera_pose_contact_graspnet = copy.deepcopy(camera_pose)
    camera_pose_contact_graspnet[0:3, 3] = camera_pose_contact_graspnet[0:3, 3] / nerf_scale
    mesh_v = np.matmul(np.linalg.inv(camera_pose_contact_graspnet), mesh_v)[:-1].T
    npy_data['xyz'] = mesh_v
    npy_data['K'] = camera_intrinsic

    np.save("./npy_data/sq_npy_data", npy_data)

    # TODO: transfer the codes here to obtain the predicted grasp poses from Contact GraspNet directly
    # Temporarily, it's achieved by reading the results from the pipeline
    predictions_data = np.load("./npy_data/predictions_sq_npz_data.npz", allow_pickle=True)
    grasp_poses = predictions_data["pred_grasps_cam"].item()[-1]

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
    



    ## Evaluate the grasp candidates and visualize them
    # Define the gripper
    scale = 5
    gripper_width = 2 * scale
    gripper_length = 1 * scale

    # Key points on the gripper
    num_sample = 10
    arm_end = np.array([0, 0, -gripper_length])
    center = np.array([0, 0, 0])
    elbow1 = np.array([gripper_width/2, 0, 0])
    elbow2 = np.array([-gripper_width/2, 0, 0])
    tip1 = np.array([gripper_width/2, 0, gripper_length])
    tip2 = np.array([-gripper_width/2, 0, gripper_length])

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
    
    


    print("Evaluating Grasp Qualities....")
    print(grasp_poses[0])
    grasp_cands = [] # All the grasp candidates
    # Construct the grasp poses at the specified locations,
    # and add them to the visualizer
    for idx in range(grasp_poses.shape[0]):
        grasp_pose = grasp_poses[idx]
        # Find the grasp pose in the world frame (converted from camera frame)
        grasp_pose = np.matmul(camera_pose_contact_graspnet, grasp_pose)
        grasp_pose[0:3, 3] = grasp_pose[0:3, 3] * nerf_scale # Convert it back to the correct scale
       
        
        # Transform the associated points for visualization or collision testing to the correct location
        gripper_points_vis = np.vstack((gripper_points.T, np.ones((1, gripper_points.shape[0]))))
        gripper_points_vis = np.matmul(grasp_pose, gripper_points_vis)
        gripper_points_vis_sample = np.vstack(\
            (gripper_points_sample.T, np.ones((1, gripper_points_sample.shape[0]))))
        gripper_points_vis_sample = np.matmul(grasp_pose, gripper_points_vis_sample)
        
        grasp_pose_lineset = o3d.geometry.LineSet()
        grasp_pose_lineset.points = o3d.utility.Vector3dVector(gripper_points_vis[:-1].T)
        grasp_pose_lineset.lines = o3d.utility.Vector2iVector(gripper_lines)
        
        antipodal_res, bbox = antipodal_test(mesh, grasp_pose, gripper_attr, 5, np.pi/9)
        # Collision Test
        if collision_test_local(mesh, gripper_points_vis_sample[:-1].T, \
                                grasp_pose, gripper_attr, threshold=0.2 * gripper_width):
            grasp_pose_lineset.paint_uniform_color((1, 0, 0))
        else:
            if antipodal_res == True:
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
    vis.add_geometry(frame)
    vis.add_geometry(camera_frame)
    vis.add_geometry(ball_select)

    for grasp_cand in grasp_cands:
        vis.add_geometry(grasp_cand)

    vis.run()
    vis.destroy_window()


