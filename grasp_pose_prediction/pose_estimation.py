import numpy as np
import os
import sys
sys.path.append(os.getcwd())

import open3d as o3d

import argparse
from superquadrics import *
from mesh_process import *
from image_process import *

from scipy.spatial.transform import Rotation as R

# Necessary Packages for LoFTR for Feature Matching
from LoFTR.src.utils.plotting import make_matching_figure
from LoFTR.src.loftr import LoFTR, default_cfg
'''
Main Program of the whole grasp pose prediction module
using Marching Primitives to split the target object into sq's
'''

# TODO: Add argument parser here
def estimate_camera_pose(parser, argv):
    args = parser.parse_args(argv)

    ######
    # Part 0: Read the Image
    ######
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
    print("====================")
    print("Image Pose to Estimate")
    print(img_file)
    # Read the ground-truth camera pose of that image
    camera_proj_img, camera_pose, nerf_scale = read_proj_from_json(img_dir, img_file)
    print("=====================")
    print("Ground-truth Camera Pose: ")
    print(camera_pose)

    ######
    ## Part I: Find matches between the sample image and the reference image
    ######
    # Specify the reference images
    images_reference_list = [
        "/images/" + "chair_0_8.png",
        "/images/" + "chair_0_16.png",
        "/images/" + "chair_0_23.png",
        "/images/" + "chair_0_29.png",
        "/images/" + "chair_1_8.png",
        "/images/" + "chair_1_16.png",
        "/images/" + "chair_1_23.png",
        "/images/" + "chair_1_29.png",
        "/images/" + "chair_2_8.png",
        "/images/" + "chair_2_16.png",
        "/images/" + "chair_2_23.png",
        "/images/" + "chair_2_29.png",
        "/images/" + "chair_3_8.png",
        "/images/" + "chair_3_16.png",
        "/images/" + "chair_3_23.png",
        "/images/" + "chair_3_29.png"
    ]
    # Build up the matcher
    image_type = 'outdoor'
    # Build up the feature matcher
    matcher = LoFTR(config=default_cfg)
    if image_type == 'indoor':
        matcher.load_state_dict(torch.load("./LoFTR/weights/indoor_ds_new.ckpt")['state_dict'])
    elif image_type == 'outdoor':
        matcher.load_state_dict(torch.load("./LoFTR/weights/outdoor_ds.ckpt")['state_dict'])
    else:
        raise ValueError("Wrong image_type is given.")
    matcher = matcher.eval().cuda()
    print("===Closest Image==")
    
    # Find the closest reference image to the the sample image w.r.t number of matches
    image_name_reference = image_select_closest(img_file, images_reference_list, \
                                img_dir, matcher)
    print(image_name_reference)
    # Determine the pixel coordinates of the point matches
    pixel_coords_img, pixel_coords_ref = match_coords(img_dir + img_file, \
                                                      img_dir + image_name_reference, \
                                matcher, th = 0.7, scale_restore=True)
    ######
    ## Part II: Estimate Camera pose of sampled image
    ######
    ## 2D-3D Matches by Raycasting on the Scene
    # Obtain the directinal vectors of the selected points
    ray_dir, camera_pose_ref, _, nerf_scale = \
        dir_point_on_image(img_dir, image_name_reference, pixel_coords_ref)

    # Specify the mesh file
    filename=img_dir + "/" + args.mesh_name
    
    # Read the file as a triangular mesh (assume that the coordinates are already corrected)
    mesh = o3d.io.read_triangle_mesh(filename)

    # Obtain the distances & positions of the selected points
    pos, dist  = point_select_in_space(camera_pose_ref, ray_dir, mesh)
    pos = pos[dist < 100, :]
    pixel_coords_img = pixel_coords_img[dist < 100, :]
    print("====Test ====")
    ## Calculate the Camera pose of the sampled image using PnP-RANSAC
    # Obtain the rotational & translational vector  
    results = cv2.solvePnPRansac(pos, pixel_coords_img, camera_proj_img[:, :-1], None)
    rvec = results[1]
    tvec = results[2]
    rot_est = cv2.Rodrigues(rvec.reshape(1, -1))[0]
    camera_pose_est = np.vstack(\
        (np.hstack((rot_est, tvec.reshape(-1,1))), np.array([0, 0, 0, 1])))
    camera_pose_est = np.linalg.inv(camera_pose_est)
    print(camera_pose_est)
    # Plot out the camera frame
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    camera_frame.scale(20/64 * nerf_scale, [0, 0, 0])
    camera_frame.transform(camera_pose)

    camera_frame_est = o3d.geometry.TriangleMesh.create_coordinate_frame()
    camera_frame_est.scale(20/64 * nerf_scale, [0, 0, 0])
    camera_frame_est.transform(camera_pose_est)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos)
    pcd.paint_uniform_color((1, 0, 0))

    ball_select =  o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    ball_select.scale(1/64 * nerf_scale, [0, 0, 0])

    ball_select.translate((pos[0][0], pos[0][1], pos[0][2]))
    ball_select.paint_uniform_color((1, 0, 0))
    print("====Test Reprojected Points ==")
    # Project all points back onto the image plane.
    # image = cv2.imread(img_dir + img_file)
    # (reproj_points, jacobian) = \
    #     cv2.projectPoints(pos, rvec, tvec, camera_proj_img[:, :-1], None)
    
    # for p in reproj_points:
    #     cv2.circle(image, (int(p[0][0]), int(p[0][1])), 3, (0,0,255), -1)
    
    # # Display image
    # cv2.imshow("Output", image)
    # cv2.waitKey(0)

    # rot = np.linalg.inv(camera_pose)[:, :-1]
    # rot = rot[:-1, :]
    # rvec2, _= cv2.Rodrigues(rot)
    # print(rot)
    # print(rvec2)
    # tvec2 = np.linalg.inv(camera_pose)[0:3, 3]

    # (reproj_points2, jacobian) = \
    #     cv2.projectPoints(pos, rvec2, tvec2, camera_proj_img[:, :-1], None)
    # image2 = cv2.imread(img_dir + img_file)
    # for p2 in reproj_points2:
    #     cv2.circle(image2, (int(p2[0][0]), int(p2[0][1])), 3, (0,0,255), -1)
    
    # # Display image
    # cv2.imshow("Output2", image2)
    # cv2.waitKey(0)

    ##########
    # Postlude: Plot out the Visualizations
    ##########
    # Create the window to display everything
    vis= o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    vis.add_geometry(camera_frame)
    vis.add_geometry(camera_frame_est)
    vis.add_geometry(pcd)
    vis.add_geometry(ball_select)
    vis.run()

    # Close all windows
    vis.destroy_window()






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
    estimate_camera_pose(parser, sys.argv[1:])