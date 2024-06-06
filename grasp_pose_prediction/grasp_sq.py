import numpy as np
import os
import sys
sys.path.append(os.getcwd() + "/superquadric_parsing")

import open3d as o3d

import argparse
from superquadrics import *
from mesh_process import *
from image_process import *

from scipy.spatial.transform import Rotation as R

# Necessary Packages for sq parsing
from scripts.train_network import sq_parsing_train
from scripts.forward_pass import sq_parsing_predict

from scripts.arguments import add_voxelizer_parameters, add_nn_parameters,\
    add_dataset_parameters, add_training_parameters,\
    add_regularizer_parameters, add_sq_mesh_sampler_parameters,\
    add_gaussian_noise_layer_parameters, voxelizer_shape,\
    add_loss_options_parameters, add_loss_parameters, get_loss_options
from scripts.arguments import add_voxelizer_parameters, add_nn_parameters, \
     add_dataset_parameters, add_gaussian_noise_layer_parameters, \
     voxelizer_shape, add_loss_options_parameters, add_loss_parameters
'''
Main Program of the whole grasp pose prediction module
'''

# TODO: Add argument parser here
def predict_grasp_pose_sq(parser, argv):
    args = parser.parse_args(argv)

    ## The image used to specify the selected point
    img_dir = args.nerf_dataset

    # If the image is not specified, select one image by random
    if args.image_name is None:
        image_files = os.listdir(args.nerf_dataset + "/images")
        image_idx = np.random.randint(0, len(image_files))
        image_name = image_files[image_idx]
    else:
        image_name = args.image_name


    img_file = "/images/" + image_name
    print(img_file)
    print("====================")
    print("Select from Image")
    ## Obtain the ray direction of the selected point in space 
    ray_dir, camera_pose, _, nerf_scale = point_select_from_image(img_dir, img_file, save_fig=True)
    ray_dir = ray_dir / np.linalg.norm(ray_dir)

    ## Create files as input to other modules
    # Specify the ply file
    filename=img_dir + "/" + args.mesh_name
    
    # Read the file as a triangular mesh
    mesh = o3d.io.read_triangle_mesh(filename)

    # Part I: create the normalized mesh as the input to superquadrics (in the format of shapenetv2)
    # Store the normalized model in the correct dataset easy to be read by sq network
    # (avoid mixing with the dataset used for NeRF)
    normalize_obj_dir = args.dataset_directory

    # Check whether the normalized mesh has already stored in the dataset
    try:
        # Check whether the dataset exists
        os.listdir(normalize_obj_dir)
        normalize_obj_dir = normalize_obj_dir + "/custom_data_1/models"
        normalize_obj = normalize_obj_dir + "/model_normalized.obj"
        os.path.isfile(normalize_obj)
    except:
        # Create the dataset if it's not existing
        # Obey the directory tree hierarchy shown in shapenet v2
        os.mkdir(normalize_obj_dir)
        normalize_obj_dir = normalize_obj_dir + "/custom_data_1"
        os.mkdir(normalize_obj_dir)
        normalize_obj_dir = normalize_obj_dir + "/models"
        os.mkdir(normalize_obj_dir)
        normalize_obj = normalize_obj_dir + "/model_normalized.obj"

    # Correct the coordinate convention if the normalized model is not generated
    if not os.path.isfile(normalize_obj): 
        # Restore the correct scale & fix up the coordinate issue 
        mesh = coordinate_correction(mesh, filename, nerf_scale)

        # Output the normalized triangular mesh & Obtain the stats
        stats = model_normalized(filename, normalize_obj, \
                                 normalize_stats_file= normalize_obj_dir + "/normalize.npy", stats = None)
    else:
        # If the normalize model already exists, read the parameters directly
        stats = read_normalize_stats(normalize_obj_dir + "/normalize.npy").item()

    # Generate the predicted superquadrics file
    if args.train:
        sq_parsing_train(parser, argv)
    sq_parsing_predict(parser, False, argv) # DO NOT CALL MLAB

    ## Determine the location of the selected point in space
    pos, point_select_distance = point_select_in_space(camera_pose, ray_dir, mesh)
    print("========================")
    print("Selected Point in Space: ")
    print("[%.2f, %.2f, %.2f]"%(pos[0], pos[1], pos[2]))
    print("========================")
    print('Visualizing...')

    # TODO: Crop a local region at the specified point





    # Add a sphere to the selected point
    ball_select =  o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    ball_select.scale(1/64 * nerf_scale, [0, 0, 0])

    ball_select.translate((pos[0], pos[1], pos[2]))
    ball_select.paint_uniform_color((1, 0, 0))

    # Plot out the camera frame
    # camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # camera_frame.scale(20/64 * nerf_scale, [0, 0, 0])
    # camera_frame.transform(camera_pose)
    

    ## Read the parameters of the superquadrics and import the point

    # Extract out the necessary attributes
    diag = stats['max'] - stats['min']
    norm = np.linalg.norm(diag)
    c = stats['centroid']
    
    # Read the attributes of the predicted sq
    sq_dir = args.predict_output_directory
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
    # sq_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # sq_frame.scale(20/64 * nerf_scale, [0, 0, 0])
    # sq_frame.transform(sq_closest["transformation"])
    print("================================")
    print("Selected superquadric Parameters: ")
    print(sq_closest["sq_parameters"])
    print(norm)


    ## Determine the grasp candidates on this sq and visualize them
    # Define the gripper
    # From experiments, when nerf_scale is 64, the gripper of length 10 looks
    # pretty nice
    scale = 5 
    gripper_width = 2 * scale * (nerf_scale/ 64)
    gripper_length = 2 * scale * (nerf_scale/ 64)
    gripper_thickness = 0.4 * gripper_length

    # Key points on the gripper
    num_sample = 20
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
    gripper_part4 = np.linspace(elbow1, elbow2, num_sample)
    gripper_points_sample = np.vstack((gripper_part1, gripper_part2, gripper_part3, gripper_part4))

    # Add the thickness
    gripper_point_sample1 = copy.deepcopy(gripper_points_sample)
    gripper_point_sample1[:, 1] = -gripper_thickness/2
    gripper_point_sample2 = copy.deepcopy(gripper_points_sample)
    gripper_point_sample2[:, 1] = gripper_thickness/2

    # Stack all points together
    gripper_points_sample = np.vstack((gripper_points_sample, gripper_point_sample1, gripper_point_sample2))


    # Pass the gripper attributes to predict the grasp poses
    gripper_attr = {"Type": "parallel", \
                    "Width": gripper_width, "Length": gripper_length, "Thickness": gripper_thickness,\
                        "Scale": scale}
    grasp_poses = grasp_pose_predict_sq_closest(sq_closest, gripper_attr, norm, sample_number=40)
    


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
        
        # Do the necessary testing jobs
        # grasp_pose_pc = o3d.geometry.PointCloud()
        # grasp_pose_pc.points = o3d.utility.Vector3dVector(gripper_points_vis_sample[:-1].T)
        # grasp_pose_pc.paint_uniform_color((1, 0, 0))
        antipodal_res, bbox = antipodal_test(mesh, grasp_pose, gripper_attr, 5, np.pi/18)
        collision_res = collision_test(mesh, gripper_points_vis_sample[:-1].T, threshold=0.02 * gripper_width)
        # Collision Test
        if collision_res:
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
    # vis.add_geometry(sq_frame)
    vis.add_geometry(frame)
    #vis.add_geometry(camera_frame)
    vis.add_geometry(ball_select)
    for grasp_cand in grasp_cands:
        vis.add_geometry(grasp_cand)
    for bbox_cand in bbox_cands:
        vis.add_geometry(bbox_cand)
    vis.run()

    # Close all windows
    vis.destroy_window()

    # Print out the validation results
    print("*******************")
    print("** Grasp pose Prediction Result: ")
    print("Selected Point in Space: ")
    print("[%.2f, %.2f, %.2f]"%(pos[0], pos[1], pos[2]))
    print("Number of valid grasp poses predicted: " + str(len(bbox_cands)))
    print("*******************")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a network to predict primitives & predict grasp poses on top of them"
    )
    ## Arguments for NeRF reconstruction stuff
    parser.add_argument(
        "nerf_dataset",
        help="The dataset containing all the training images & transform.json"
    )
    parser.add_argument(
        "--image_name",
        help="The name of the image to use for point selection"
    )
    parser.add_argument(
        "--mesh_name",
        default = "chair_upper.obj",
        help="The name of the mesh model to use"
    )
    parser.add_argument(
        "--dataset_directory",
        default="./superquadric_parsing/dataset/custom_data_upper",
        help="Path to the directory containing the dataset"
    )
    parser.add_argument(
        "--train_output_directory",
        default="./superquadric_parsing/sq_parsing_train_results",
        help="Save the output files from training process in that directory"
    )
    parser.add_argument(
        "--predict_output_directory",
        default="./superquadric_parsing/sq_parsing_predict_results",
        help="Save the output files from prediction process in that directory"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Whether to train the superquadric parsing network from the beginning again"
    )
    parser.add_argument(
        "--tsdf_directory",
        default="",
        help="Path to the directory containing the precomputed tsdf files"
    )

    ## Arguments for training process
    parser.add_argument(
        "--train_weight_file",
        default=None,
        help=("The path to a previously trainined model to continue"
              " the training from")
    )
    parser.add_argument(
        "--continue_from_epoch",
        default=0,
        type=int,
        help="Continue training from epoch (default=0)"
    )
    parser.add_argument(
        "--n_primitives",
        type=int,
        default=32,
        help="Number of primitives"
    )
    parser.add_argument(
        "--use_deformations",
        action="store_true",
        help="Use Superquadrics with deformations as the shape configuration"
    )
    parser.add_argument(
        "--train_test_splits_file",
        default=None,
        help="Path to the train-test splits file"
    )
    parser.add_argument(
        "--run_on_gpu",
        action="store_true",
        help="Use GPU"
    )
    parser.add_argument(
        "--probs_only",
        action="store_true",
        help="Optimize only using the probabilities"
    )

    parser.add_argument(
        "--experiment_tag",
        default=None,
        help="Tag that refers to the current experiment"
    )

    parser.add_argument(
        "--cache_size",
        type=int,
        default=2000,
        help="The batch provider cache size"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=27,
        help="Seed for the PRNG"
    )
    
    ## Arguments for prediction process
    parser.add_argument(
        "--predict_weight_file",
        default=None,
        help="The path to the previously trainined model to be used"
    )

    # Same number of primitives as in training
    # parser.add_argument(
    #     "--n_primitives",
    #     type=int,
    #     default=32,
    #     help="Number of primitives"
    # )

    parser.add_argument(
        "--prob_threshold",
        type=float,
        default=0.5,
        help="Probability threshold"
    )

    # Obey the rule in training in whether to use deformations
    # parser.add_argument(
    #     "--use_deformations",
    #     action="store_true",
    #     help="Use Superquadrics with deformations as the shape configuration"
    # )
    parser.add_argument(
        "--save_prediction_as_mesh",
        action="store_true",
        help="When true store prediction as a mesh"
    )
    add_nn_parameters(parser)
    add_dataset_parameters(parser)
    add_voxelizer_parameters(parser)
    add_training_parameters(parser)
    add_sq_mesh_sampler_parameters(parser)
    add_regularizer_parameters(parser)
    add_gaussian_noise_layer_parameters(parser)
    # Parameters related to the loss function and the loss weights
    add_loss_parameters(parser)
    # Parameters related to loss options
    add_loss_options_parameters(parser)

    predict_grasp_pose_sq(parser, sys.argv[1:])