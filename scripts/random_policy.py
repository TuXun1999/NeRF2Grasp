# Main loop of the Uncertainty-Guided Policy
from UGP_pipeline import UGP_pipeline
import argparse
import os
import commentjson as json

import numpy as np
from scipy.spatial.transform import Rotation as R
def parse_args():
    parser = argparse.ArgumentParser(description="Run instant neural graphics primitives with additional configuration & output options")

    parser.add_argument("files", nargs="*", help="Files to be loaded. Can be a scene, network config, snapshot, camera path, or a combination of those.")

    parser.add_argument("--scene", "--training_data", default="", help="The scene to load. Can be the scene's name or a full path to the training data. Can be NeRF dataset, a *.obj/*.stl mesh for training a SDF, an image, or a *.nvdb volume.")
    parser.add_argument("--mode", default="", type=str, help=argparse.SUPPRESS) # deprecated
    parser.add_argument("--network", default="", help="Path to the network config. Uses the scene's default if unspecified.")

    parser.add_argument("--load_snapshot", "--snapshot", default="", help="Load this snapshot before training. recommended extension: .ingp/.msgpack")
    parser.add_argument("--save_snapshot", default="", help="Save this snapshot after training. recommended extension: .ingp/.msgpack")

    parser.add_argument("--nerf_compatibility", action="store_true", help="Matches parameters with original NeRF. Can cause slowness and worse results on some scenes, but helps with high PSNR on synthetic scenes.")
    parser.add_argument("--test_transforms", default="", help="Path to a nerf style transforms json from which we will compute PSNR.")
    parser.add_argument("--near_distance", default=-1, type=float, help="Set the distance from the camera at which training rays start for nerf. <0 means use ngp default")
    parser.add_argument("--exposure", default=0.0, type=float, help="Controls the brightness of the image. Positive numbers increase brightness, negative numbers decrease it.")

    parser.add_argument("--screenshot_transforms", default="", help="Path to a nerf style transforms.json from which to save screenshots.")
    parser.add_argument("--screenshot_frames", nargs="*", help="Which frame(s) to take screenshots of.")
    parser.add_argument("--screenshot_dir", default="", help="Which directory to output screenshots to.")
    parser.add_argument("--screenshot_spp", type=int, default=16, help="Number of samples per pixel in screenshots.")

    parser.add_argument("--video_camera_path", default="", help="The camera path to render, e.g., base_cam.json.")
    parser.add_argument("--video_camera_smoothing", action="store_true", help="Applies additional smoothing to the camera trajectory with the caveat that the endpoint of the camera path may not be reached.")
    parser.add_argument("--video_fps", type=int, default=60, help="Number of frames per second.")
    parser.add_argument("--video_n_seconds", type=int, default=1, help="Number of seconds the rendered video should be long.")
    parser.add_argument("--video_render_range", type=int, nargs=2, default=(-1, -1), metavar=("START_FRAME", "END_FRAME"), help="Limit output to frames between START_FRAME and END_FRAME (inclusive)")
    parser.add_argument("--video_spp", type=int, default=8, help="Number of samples per pixel. A larger number means less noise, but slower rendering.")
    parser.add_argument("--video_output", type=str, default="video.mp4", help="Filename of the output video (video.mp4) or video frames (video_%%04d.png).")

    parser.add_argument("--save_mesh", default="", help="Output a marching-cubes based mesh from the NeRF or SDF model. Supports OBJ and PLY format.")
    parser.add_argument("--marching_cubes_res", default=256, type=int, help="Sets the resolution for the marching cubes grid.")
    parser.add_argument("--marching_cubes_density_thresh", default=2.5, type=float, help="Sets the density threshold for marching cubes.")

    parser.add_argument("--width", "--screenshot_w", type=int, default=0, help="Resolution width of GUI and screenshots.")
    parser.add_argument("--height", "--screenshot_h", type=int, default=0, help="Resolution height of GUI and screenshots.")

    parser.add_argument("--gui", action="store_true", help="Run the testbed GUI interactively.")
    parser.add_argument("--train", action="store_true", help="If the GUI is enabled, controls whether training starts immediately.")
    parser.add_argument("--n_steps", type=int, default=-1, help="Number of steps to train for before quitting.")
    parser.add_argument("--second_window", action="store_true", help="Open a second window containing a copy of the main output.")
    parser.add_argument("--vr", action="store_true", help="Render to a VR headset.")
    parser.add_argument("--depth_supervision_lambda", type=float, default=0.1, help="Depth supervision lambda to enable training with depth")
    parser.add_argument("--sharpen", default=0, help="Set amount of sharpening applied to NeRF training images. Range 0.0 to 1.0.")
    parser.add_argument("--ugp_debug", action="store_true", help="Output debug messages for UGP")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    testbed_args = args
    pyrender_args = {}
    pyrender_args["obj"]= 'pyrender/examples/models/chair1.obj'
    pyrender_args["dataset_dir"] = args.scene
    ugp_pipeline = UGP_pipeline(pyrender_args = pyrender_args)

    ## Take six images as the initialization
    # TODO: complete the whole UGP process

    # Initialize the camera poses
    # Initial rotations to adjust the pose of the camera

    # Rotate around the x-axis for pi/2
    rot1 = R.from_quat([np.sin(np.pi/4), 0, 0, np.cos(np.pi/4)]).as_matrix()

    # Rotate around the z-axis for pi
    rot2 = R.from_quat([0, 0, np.sin(np.pi/2), np.cos(np.pi/2)]).as_matrix()

    # Combine the two rotations
    rot = np.matmul(rot2, rot1)

    # Translation part
    d = np.array([[0], [125], [0]])

    # Transformation matrix at the initial pose
    trans_initial = np.vstack((np.hstack((rot, d)), np.array([0, 0, 0, 1])))

    # Number of samples vertically
    h_num = 5
    # Number of samples along the circle around the object
    i_num = 30
    angle_dev_i = 2*np.pi/i_num
    angle_dev_h = (np.pi/2)/h_num

    # Initialize all the camera view pose candidates
    # Establish the original whole view space
    camera_pose_candidates = []

    for h in range(h_num): # Rotate around axis-x
        rot_angle_h = h * angle_dev_h
        # Rotation matrix for the camera pose around x-axis
        rot_around_x = R.from_quat([np.sin(rot_angle_h/2), 0, 0, np.cos(rot_angle_h/2)]).as_matrix()
        camera_pose_vet = np.matmul(
                np.vstack(
                    (np.hstack((rot_around_x, np.array([[0], [0], [0]]))), 
                    np.array([0, 0, 0, 1]))),
                trans_initial
                )
        for i in range(i_num): # Rotate around axis-z
            rot_angle_i  = i * angle_dev_i

            # Rotation matrix for that camera pose
            rot_around_z = R.from_quat([0, 0, np.sin(rot_angle_i/2), np.cos(rot_angle_i/2)]).as_matrix()
            # Move camera to that sampling pose
            camera_pose = {}
            camera_pose["transformation"] = np.matmul(
                np.vstack(
                    (np.hstack((rot_around_z, np.array([[0], [0], [0]]))), 
                    np.array([0, 0, 0, 1]))),
                camera_pose_vet
                )
            camera_pose["image_name"]= str(h) + "_" + str(i) + ".png"
            camera_pose["selected"] = False
            camera_pose["fov"] = 90
            camera_pose["render_image_name"] = str(h) + "_" + str(i) + "_render.png"
            camera_pose_candidates.append(camera_pose)

    iter_num = 4
    pose_numbers = [6, 18, 30, 42]

    # Randomly select some poses
    old_pose_num = 0
    pose_idx_selected = np.random.choice(range(len(camera_pose_candidates)), pose_numbers[-1], replace=False)
    for pose_num in pose_numbers:
        # If requested, output a reference image at the start of each iteration
        
        camera_pose_idx = list(pose_idx_selected[old_pose_num:pose_num])
        old_pose_num = pose_num
        print(camera_pose_idx)
        # Select new poses and take snapshots at those poses
        new_poses = [camera_pose_candidates[i] for i in camera_pose_idx]
        # Take the new 12 images and add them to the dataset
        ugp_pipeline.pyrender_take_snapshot(new_poses, pyrender_args)

        # Use the new dataset to re-train the nerf model
        ugp_pipeline.init_testbed(testbed_args=testbed_args)
        if (pose_num == 6):
            ugp_pipeline.train_NeRF(testbed_args, start_from_prev_snapshot=False)
        else:
            ugp_pipeline.train_NeRF(testbed_args, start_from_prev_snapshot=True)
        if testbed_args.ugp_debug:
            # Take a snapshot at a reference pose
            entropy_poses = []
            entropy_pose1 = {}
            entropy_pose1["transformation"] = np.matrix([
                [
                    0.8090169943749476,
                    0.3454915028125266,
                    -0.4755282581475771,
                    -0.9287661291944864
                ],
                [
                    -0.5877852522924736,
                    0.4755282581475771,
                    -0.6545084971874736,
                    -1.2783369085692848
                ],
                [
                    7.198293278059969e-17,
                    0.8090169943749477,
                    0.5877852522924736,
                    1.148018070883737
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ])
            entropy_pose1["render_image_name"] = "render_test_" + str(iter) + ".png"
            entropy_pose1["fov"] = 90
            entropy_poses.append(entropy_pose1)
            # Look at the entropy image as a reference
            ugp_pipeline.get_render_image(entropy_poses, mode = "RGB")
            ugp_pipeline.get_render_image(entropy_poses, mode = "entropy")
