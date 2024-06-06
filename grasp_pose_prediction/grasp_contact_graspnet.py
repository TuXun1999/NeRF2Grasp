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
'''
Main Program of the baseline evaluation Module (Contact GraspNet)
'''


# TODO: Add argument parser here
def contact_graspnet_benchmark(global_config, 
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

    camera_pose_contact_graspnet = copy.deepcopy(camera_pose)
    camera_pose_contact_graspnet[0:3, 3] = camera_pose_contact_graspnet[0:3, 3] / nerf_scale
    # Create the npy file as the input to Contact GraspNet

    npy_data = {}
    # In order that Contact GraspNet can work normally, we have to scale the model down
    npy_data['xyz_color'] = np.asarray(mesh.vertex_colors)

    mesh_v = np.asarray(mesh.vertices) / nerf_scale
    mesh_v = np.vstack((mesh_v.T, np.ones((1, mesh_v.shape[0]))))
    
    mesh_v = np.matmul(np.linalg.inv(camera_pose_contact_graspnet), mesh_v)[:-1].T
    npy_data['xyz'] = mesh_v
    npy_data['K'] = camera_intrinsic

    np.save(input_paths + "/cg_input.npy", npy_data)

    # Transfer the codes here to obtain the predicted grasp poses from Contact GraspNet directly
    grasp_cand_sample = 20 * points_in_space.shape[0]
    grasp_poses, grasp_scores = inference(global_config, 
              ckpt_dir,
              input_paths + "/cg_input.npy", 
              local_regions, 
              filter_grasps, 
              skip_border_objects,
              z_range,
              forward_passes,
              K)

    grasp_poses = grasp_poses[-1]
    grasp_scores = grasp_scores[-1]
    
    # Extract out the 20 grasp poses with the maximum scores
    grasp_max_idx = np.argpartition(grasp_scores, -grasp_cand_sample)[-grasp_cand_sample:]
    grasp_poses = grasp_poses[grasp_max_idx]
    # grasp_poses = np.load("./contact_graspnet_results/predictions_sq_npz_data.npz", allow_pickle=True)
    # grasp_poses = grasp_poses["pred_grasps_cam"].item()[-1]
    # print(grasp_poses[0])

    ## Determine the location of the selected point in space
    pos, point_select_distance = point_select_in_space(camera_pose, ray_dir, mesh)
    print(point_select_distance)
    print('Visualizing...')

    # TODO: Crop a local region at the specified point


    # Add a sphere to the selected point
    # ball_select =  o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    # ball_select.scale(1/64 * nerf_scale, [0, 0, 0])

    # ball_select.translate((pos[0], pos[1], pos[2]))
    # ball_select.paint_uniform_color((1, 0, 0))

    # Plot out the camera frame
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
        
        antipodal_res, bbox = antipodal_test(mesh, grasp_pose, gripper_attr, 5, np.pi/18)
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
        
    ## Postlogue
    # Plot out the fundamental frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    frame.scale(20/64 * nerf_scale, [0, 0, 0])


                                                                                                                                                                    
    # Create the window to display everything
    vis= o3d.visualization.Visualizer()

    # vis_geoms = [mesh, frame, camera_frame]
    vis.create_window()
    vis.add_geometry(mesh)
    vis.add_geometry(frame)
    vis.add_geometry(camera_frame)

    for grasp_cand in grasp_cands:
        vis.add_geometry(grasp_cand)
        # vis_geoms.append(grasp_cand)

    # Plot out the point candidates in space
    # mat_point_cand = o3d.visualization.rendering.MaterialRecord()
    # # mat_box.shader = 'defaultLitTransparency'
    # mat_point_cand.shader = 'defaultLitSSR'
    # mat_point_cand.base_color = [0.467, 0.467, 0.467, 0.2]
    # mat_point_cand.base_roughness = 0.0
    # mat_point_cand.base_reflectance = 0.0
    # mat_point_cand.base_clearcoat = 1.0
    # mat_point_cand.thickness = 1.0
    # mat_point_cand.transmission = 0.2
    # mat_point_cand.absorption_distance = 10
    # mat_point_cand.absorption_color = [0.5, 0.5, 0.5]

    # Construct the geometry
    point_idx = 0
    for point_in_space in points_in_space:
        point_sample_select =  o3d.geometry.TriangleMesh.create_sphere(radius=3.0, resolution=20)
        point_sample_select.scale(1/64 * nerf_scale, [0, 0, 0])

        point_sample_select.translate((point_in_space[0], point_in_space[1], point_in_space[2]))
        point_sample_select.paint_uniform_color([1, 0, 0])
        # point_geom =  o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
        # point_geom.scale(10/64 * nerf_scale, [0, 0, 0])

        # point_geom.translate((point_in_space[0], point_in_space[1], point_in_space[2]))
        # point_geom.paint_uniform_color((1, 0, 0))

        # point_draw = {'name': 'point_sample' + "_" + str(point_idx), 'geometry': point_geom, 'material': mat_point_cand}

        # vis_geoms.append(point_draw)
        vis.add_geometry(point_sample_select)
        point_idx += 1
    
    vis.run()
    vis.destroy_window()
    # o3d.visualization.draw(vis_geoms)


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
    contact_graspnet_benchmark(global_config, 
            FLAGS.ckpt_dir,
            local_regions=FLAGS.local_regions,
            filter_grasps=FLAGS.filter_grasps,
            skip_border_objects=FLAGS.skip_border_objects,
            z_range=eval(str(FLAGS.z_range)),
            forward_passes=FLAGS.forward_passes,
            K=eval(str(FLAGS.K)))
