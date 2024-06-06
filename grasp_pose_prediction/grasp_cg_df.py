import numpy as np

import copy
import open3d as o3d
from superquadrics import *
from mesh_process import *
from image_process import *
import os
import sys
sys.path.append(os.getcwd())
from scipy.spatial.transform import Rotation as R


import argparse

import numpy as np

from cg_baseline import config_utils
from cg_baseline.inference import inference

from darboux_frame import fit_neighbor, calculate_darboux_frame
from sklearn.neighbors import KDTree
'''
Main Program of the baseline evaluation Module 
(Darboux Frame at the local region + Contact GraspNet)
'''


# TODO: Add argument parser here
def contact_graspnet_df_benchmark(global_config, 
              ckpt_dir,
              local_regions=True, 
              filter_grasps=True, 
              skip_border_objects=False,
              z_range = [0.2,1.8],
              forward_passes=1,
              K=None,):

    ## The image used to specify the selected point
    img_dir = "./data/nerf/chair10_pm"
    img_file = "/images/chair_1_23.png"

    # Create the window to display everything
    vis= o3d.visualization.Visualizer()


    # Specify the directory to store the npy data
    input_paths = "./npy_data/" + img_dir.split('/')[-1]
    try:
        os.listdir(input_paths)
    except:
        os.mkdir(input_paths)
    # Create the point candidate in space
    points_in_space = np.array([[0.07, 0.45, 0.38],
                                [0.13, -0.49, 0.29],
                                [0.44, 0.28, 0.53],
                                [0.45, -0.29, 0.58]])
    ## Obtain the ray direction of the selected point in space 
    ray_dir, camera_pose, camera_intrinsic, nerf_scale = point_select_from_image(img_dir, img_file)
    ray_dir = ray_dir / np.linalg.norm(ray_dir)

    ## Create files as input to other modules
    # Specify the ply file
    filename=img_dir + "/chair_upper.obj"
    
    # Read the file as a triangular mesh
    mesh = o3d.io.read_triangle_mesh(filename)
    mesh.compute_vertex_normals()

    camera_pose_global = copy.deepcopy(camera_pose)
    camera_pose_global[0:3, 3] = camera_pose_global[0:3, 3] / nerf_scale
    # Create the npy file as the input to Contact GraspNet



    # TODO: Crop out local regions at each selected point, and feed them into contact graspnet
    grasp_poses = []
    pc = np.array(mesh.vertices) / nerf_scale
    # pc_normals = np.array(mesh.vertex_normals) # Normals for reference
    pc_tree = KDTree(pc)
    p_neighbor_th = 0.15
    camera_pose_cg = []  # The list to store the pose of all local cameras watching the regions
    camera_frame_cg = [] # The list to store the frame mesh of all local cameras watching the regions
    for point_idx in range(points_in_space.shape[0]):
        p_sel = points_in_space[point_idx] / nerf_scale
        # Use KDTree to find the points neighboring to the selected point
        p_sel_neighbor_idx = pc_tree.query_radius(p_sel.reshape(1, -1), r=p_neighbor_th)[0]

        # Abandon the current local region if there are too few points inside
        if p_sel_neighbor_idx.shape[0] < 16:
            continue 
        # The points in the local region
        p_sel_neighbor = pc[p_sel_neighbor_idx]

        # Fit the neighboring region to a quadratic surface -> in order to find 
        # the directions with principal curvatures
        c = fit_neighbor(pc, p_sel_neighbor_idx, verbose = False)
        
        # Extract out the normal at the selected point
        # When calculating the darboux frame, there will be two solutions
        # We always want the vector pointing outward from the mesh surface
        # and the analytical method will try to predict the one pointing outward
        # _, ind = pc_tree.query(p_sel.reshape(1, -1), k=1)
        # p_sel_normal = pc_normals[ind[0][0]] # Specify the general direction of the normal vector
        df_axis_1, df_axis_2, p_sel_N_curvature, k1 , k2 , quadratic = \
                calculate_darboux_frame(c, p_sel, reference_normal = None)
        # Mark the local neighboring region in a separate uniform color
        region_color = [0, 1/points_in_space.shape[0] * point_idx, 1 - point_idx / points_in_space.shape[0]]
        for idx in p_sel_neighbor_idx:
            mesh.vertex_colors[idx] = region_color

        # Apply a transformation to fit the gripper to the darboux frame
        darboux_frame_rot = np.transpose(np.vstack((df_axis_1, df_axis_2, p_sel_N_curvature)))
        darboux_frame_tran = np.array([
            [p_sel[0]],
            [p_sel[1]],
            [p_sel[2]]
        ])
        # Convert it into a standard transformation matrix
        darboux_frame = np.vstack((np.hstack((darboux_frame_rot, darboux_frame_tran)), np.array([0, 0, 0, 1])))

        # Place the camera within the local darboux frame
        # Initially, suppose the camera frame is the same as the local darboux frame
        # Apply these changes:
        # 1. Rotate around y-axis (for pi)
        # 2. Elevate the gripper for 0.5 (in the normalized scene)
        df_tran1 = R.from_quat([0, np.sin(np.pi/2), 0, np.cos(np.pi/2)]).as_matrix()
        df_tran1 = np.vstack((np.hstack((df_tran1, np.array([[0],[0],[0]]))), np.array([0, 0, 0, 1])))
        df_tran2 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0.5],
            [0, 0, 0, 1]
        ])
        # Obtain the complete local change in darboux frame
        df_tran_local = np.matmul(df_tran2, df_tran1)

        # Obtain the camera pose in world frame
        camera_pose_local = np.matmul(darboux_frame, df_tran_local)
        camera_pose_cg.append(camera_pose_local)

        # Now, construct the input to feed into contact graspnet
        npy_data = {}
        npy_data['xyz_color'] = np.asarray(mesh.vertex_colors)

        # Extract out the local neighboring region & Transform it into Homogeneous coordinate
        p_sel_neighbor = np.vstack((p_sel_neighbor.T, np.ones((1, p_sel_neighbor.shape[0]))))

        # Find the coordinates of all points in the local camera frame
        p_sel_neighbor = np.matmul(np.linalg.inv(camera_pose_local), p_sel_neighbor)[:-1].T

        # Store the data
        npy_data['xyz'] = p_sel_neighbor
        npy_data['K'] = camera_intrinsic

        np.save(input_paths + "/local_region_" + str(point_idx) + "_cg_input.npy", npy_data)

        # Transfer the codes here to obtain the predicted grasp poses from Contact GraspNet directly
        print("====================================================")
        print("Predicting Grasp poses on region " + str(point_idx))
        grasp_cand_sample = 20
        grasp_poses_local, grasp_scores_local = inference(global_config, 
                ckpt_dir,
                input_paths + "/local_region_" + str(point_idx) + "_cg_input.npy", 
                local_regions, 
                filter_grasps, 
                skip_border_objects,
                z_range,
                forward_passes,
                K)

        grasp_poses_local = grasp_poses_local[-1]
        grasp_scores_local = grasp_scores_local[-1]
        
        # Extract out the 20 grasp poses with the maximum scores
        grasp_max_idx = np.argpartition(grasp_scores_local, -grasp_cand_sample)[-grasp_cand_sample:]
        grasp_poses_local = grasp_poses_local[grasp_max_idx]

        # Append the grasp poses on the local region to the global record
        grasp_poses.append(grasp_poses_local)

    # TODO: Crop a local region at the specified point

    # Plot out the global camera frame
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    camera_frame.scale(20/64 * nerf_scale, [0, 0, 0])
    camera_frame.transform(camera_pose)
    

    ## Evaluate the grasp candidates and visualize them
    # Define the gripper
    # From experiments, when nerf_scale is 64, the gripper of length 10 looks
    # pretty nice
    scale = 5 
    gripper_width = 2 * scale * (nerf_scale/ 64)
    gripper_length = 2 * scale * (nerf_scale/ 64)
    gripper_thickness = 0.4 * gripper_length


    # Key points on the gripper
    num_sample = 20
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
    gripper_part2 = np.linspace(elbow1, tip1, 2*num_sample)
    gripper_part3 = np.linspace(elbow2, tip2, 2*num_sample)
    gripper_part4 = np.linspace(elbow1, elbow2, num_sample)
    gripper_points_sample = np.vstack((gripper_part1, gripper_part2, gripper_part3, gripper_part4))
    # Pass the gripper attributes to predict the grasp poses
    gripper_attr = {"Type": "parallel", \
                    "Width": gripper_width, "Length": gripper_length, "Thickness": gripper_thickness,\
                        "Scale": scale}
    
    


    print("Evaluating Grasp Qualities....")
    grasp_cands = [] # All the grasp candidates
    # Construct the grasp poses at the specified locations,
    # and add them to the visualizer
    for idx in range(len(grasp_poses)):
        # Evaluate the grasp pose quality on each region separately
        for grasp_idx in range(grasp_poses[idx].shape[0]):
            grasp_pose = grasp_poses[idx][grasp_idx]
            # Find the grasp pose in the world frame (converted from local camera frame)
            grasp_pose = np.matmul(camera_pose_cg[idx], grasp_pose)
            grasp_pose[0:3, 3] = grasp_pose[0:3, 3] * nerf_scale # Convert it back to the correct scale
        
            
            # Transform the associated points for visualization or collision testing to the correct location
            gripper_points_vis = np.vstack((gripper_points.T, np.ones((1, gripper_points.shape[0]))))
            gripper_points_vis = np.matmul(grasp_pose, gripper_points_vis)
            gripper_points_vis_sample = np.vstack(\
                (gripper_points_sample.T, np.ones((1, gripper_points_sample.shape[0]))))
            gripper_points_vis_sample = np.matmul(grasp_pose, gripper_points_vis_sample)
            
            # Specify the visual of the grasp pose
            grasp_pose_lineset = o3d.geometry.LineSet()
            grasp_pose_lineset.points = o3d.utility.Vector3dVector(gripper_points_vis[:-1].T)
            grasp_pose_lineset.lines = o3d.utility.Vector2iVector(gripper_lines)
            
            # Evaluate the grasp quality
            antipodal_res, _ = antipodal_test(mesh, grasp_pose, gripper_attr, 5, np.pi/18)
            collision_res = collision_test(mesh, gripper_points_vis_sample[:-1].T, threshold=0.02 * gripper_width)
            # Collision Test
            if collision_res:
                grasp_pose_lineset.paint_uniform_color((1, 0, 0))
            else:
                if antipodal_res == True:
                    grasp_pose_lineset.paint_uniform_color((0, 1, 0))
                else:
                    grasp_pose_lineset.paint_uniform_color((1, 1, 0))
            grasp_cands.append(grasp_pose_lineset)
        
        # Attach the local camera frame for visualization purpose
        camera_pose_local = camera_pose_cg[idx]
        # Also, scale the camera pose back to the correct value
        camera_pose_local[0:3, 3] = camera_pose_local[0:3, 3] * nerf_scale
        camera_frame_local = o3d.geometry.TriangleMesh.create_coordinate_frame()
        camera_frame_local.scale(5/64 * nerf_scale, [0, 0, 0])
        camera_frame_local.transform(camera_pose_local)
        camera_frame_local.paint_uniform_color(\
            [0, 1/points_in_space.shape[0] * idx, 1 - idx / points_in_space.shape[0]])
        camera_frame_cg.append(camera_frame_local)
    ## Postlogue
    # Plot out the fundamental frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    frame.scale(20/64 * nerf_scale, [0, 0, 0])

    vis.create_window()
    # Plot out the data we have collected
    vis.add_geometry(mesh)
    vis.add_geometry(frame)
    vis.add_geometry(camera_frame)

    for camera_frame_local in camera_frame_cg:
        vis.add_geometry(camera_frame_local)

    for grasp_cand in grasp_cands:
        vis.add_geometry(grasp_cand)

    # Construct the geometry of the selected points
    point_idx = 0
    for point_in_space in points_in_space:
        point_sample_select =  o3d.geometry.TriangleMesh.create_sphere(radius=3.0, resolution=20)
        point_sample_select.scale(1/64 * nerf_scale, [0, 0, 0])

        point_sample_select.translate((point_in_space[0], point_in_space[1], point_in_space[2]))
        point_sample_select.paint_uniform_color([1, 0, 0])
        vis.add_geometry(point_sample_select)
        point_idx += 1
    
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default='./cg_baseline/cg_checkpoints', help='Log dir')
    # parser.add_argument('--np_path', default='./npy_data/sq_npy_data.npy', help='Input data: npz/npy file with keys either "depth" & camera matrix "K" or just point cloud "pc" in meters. Optionally, a 2D "segmap"')
    parser.add_argument('--K', default=None, help='Flat Camera Matrix, pass as "[fx, 0, cx, 0, fy, cy, 0, 0 ,1]"')
    parser.add_argument('--z_range', default=[0.2,1.8], help='Z value threshold to crop the input point cloud')
    parser.add_argument('--local_regions', action='store_true', default=False, help='Crop 3D local regions around given segments.')
    parser.add_argument('--filter_grasps', action='store_true', default=False,  help='Filter grasp contacts according to segmap.')
    parser.add_argument('--skip_border_objects', action='store_true', default=False,  help='When extracting local_regions, ignore segments at depth map boundary.')
    parser.add_argument('--forward_passes', type=int, default=5,  help='Run multiple parallel forward passes to mesh_utils more potential contact points.')
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
    FLAGS = parser.parse_args()

    global_config = config_utils.load_config(FLAGS.ckpt_dir, batch_size=FLAGS.forward_passes, arg_configs=FLAGS.arg_configs)
    
    print(str(global_config))
    print('pid: %s'%(str(os.getpid())))
    contact_graspnet_df_benchmark(global_config, 
            FLAGS.ckpt_dir,
            local_regions=FLAGS.local_regions,
            filter_grasps=FLAGS.filter_grasps,
            skip_border_objects=FLAGS.skip_border_objects,
            z_range=eval(str(FLAGS.z_range)),
            forward_passes=FLAGS.forward_passes,
            K=eval(str(FLAGS.K)))
