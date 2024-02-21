#!/usr/bin/env python3
#
# Implementation of the Uncertainty-Guided Policy Pipeline
# Borrowed from codes in instant-ngp
# 
# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# 
#
import argparse
import os
import commentjson as json

import numpy as np

import shutil
import time

from common import *
from scenes import *

from tqdm import tqdm

import pyngp as ngp # noqa

import os
import shutil
import trimesh
import pyrender
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from PIL import Image
import json

class UGP_pipeline():
    """
    Class to implement the Uncertainty-Guided Policy for 3D scene construction using NeRF
    Paper: https://arxiv.org/pdf/2209.08409.pdf

    Consist of the four separate stages in the loop mentioned in the paper:
    1) Robot Data Acquisition: achieved by using tools in Pyrender to collect images at given poses
    2) NeRF model Training: achieved by establishing a testbed from instant-ngp and train the NeRF model
        ==> training with instant-ngp is much faster than the original NeRF pipeline
    3) Uncertainty Estimation: achieved by taking a snapshot at given poses
        ==> a small, new feature embedded into the testbed from instant-ngp to collect entropy information
    4) Next-Best-Policy Selection: achieved by replicating the steps mentioned in the paper
        ==> TODO: try to improve the policy; maybe with grasping information

    Main Attributes:
        pyrender_scene: The simulated environment to collect images from

        testbed: The platform where the NeRF model is trained, and predicted images from 
                the reconstructed 3D scene are collected
    """
    def __init__(self, pyrender_args, testbed_args = None):
        self.init_pyrender(pyrender_args=pyrender_args)
        if (testbed_args != None):
            self.init_testbed(testbed_args=testbed_args)
        
    def init_pyrender(self, pyrender_args):
        '''
        The function to initialize a simulation scene
        Input: the arguments to configure the simulation environment
        '''
        # Add the mesh
        obj_name = pyrender_args["obj"]
        fuze_trimesh = trimesh.load(obj_name)


        mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
        self.scene = pyrender.Scene(bg_color=np.array([0.0, 0.0, 0.0, 0.0]))

        # Add the mesh into the scene
        mesh_pose = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, -50],
            [0, 0, 0, 1]
        ])
        self.scene.add(mesh, pose = mesh_pose)

        # Add the lighting source
        self.light = pyrender.SpotLight(color=np.ones(3), intensity=200000.0)

        self.scene.add(self.light, pose=np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 150],
            [0, 0, 0, 1]
        ]))

        ## More light sources
        # Initial rotations to add more light sources

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

        # More lighting sources
        for l in range(4):
                light_rot = R.from_quat([0, 0, np.sin(l * np.pi/4), np.cos(l * np.pi/4)]).as_matrix()
                light_tran = np.vstack((np.hstack((light_rot, np.array([[0], [0], [0]]))), np.array([0, 0, 0, 1])))
                light_pose = np.matmul(light_tran, trans_initial)
                self.scene.add(self.light, pose=light_pose)


        self.camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.0, aspectRatio=1.0)

        # Create the dataset to store the images
        # Create the folder if it doesn't exist
        # Delete the existing one because it's not updated
        dataset_location = pyrender_args["dataset_dir"]
        image_dir = "/images/"

        if hasattr(self, 'dataset_location') == False:
            self.dataset_location = dataset_location
        
        if os.path.exists(dataset_location + image_dir) == True:
            shutil.rmtree(dataset_location + image_dir)
        
        # Re-intialize the folder.
        os.makedirs(dataset_location + image_dir)

        # Same for the rendered images
        # Path to folder to save rendered photos in.
        folder = self.dataset_location + "/rendered_images_folder"
        
        # Create the folder if it doesn't exist.
        # Remove the old ones
        if os.path.exists(folder) == True:
            shutil.rmtree(folder)
        # Intialize the folder.
        os.makedirs(folder)

    def pyrender_take_snapshot(self, poses, pyrender_args):
        '''
        The function to take snapshots on the existing simulation scene at the given poses
        Input: new poses where the cameras take snapshots
            -poses: array of class "pose"
            -pose:
                - transformation matrix
                - image_name (avoid duplicate image names)
        Output: 
            - Images taken by the cameras
            - transformation.json file used by the NeRF training testbed to train the model
        '''
        dataset_location = pyrender_args["dataset_dir"]
        
        image_dir = "/images/"

        # Create the folder if it doesn't exist
        # Add new images to the existing dataset in the following iterations
        if os.path.exists(dataset_location + image_dir) == False:
            # Should be created in the initializer
            print("Error! Did you forget to initialize the dataset?")
            return

        camera_pose_dir = dataset_location + "/transforms.json"
        # Configure the json file
        camera_pose_file = open(camera_pose_dir, "w")

        # If the camera pose records are not created, create one
        if hasattr(self, 'result_dict') == False:
            # Determine the intrinsic parameters of the camera
            self.result_dict = {}
            self.result_dict["w"] = 800
            self.result_dict["h"] = 800
            self.result_dict["aabb_scale"] = 2
            self.result_dict["fl_x"] = 400
            self.result_dict["k1"] = 0
            self.result_dict["p1"] = 0
            self.result_dict["fl_y"] = 400
            self.result_dict["k2"] = 0
            self.result_dict["p2"] = 0
            self.result_dict["cx"] = 400
            self.result_dict["cy"] = 400
            self.result_dict["camera_angle_x"] = np.pi/2
            self.result_dict["camera_angle_y"] = np.pi/2
            self.result_dict["enable_depth_loading"] = True
            self.result_dict["integer_depth_scale"] = 300/(64 * 65535)
            self.result_dict["frames"] = []
        
        for pose in poses:
            # Move camera to that sampling image_depth_napose
            camera_pose = pose["transformation"]
            nc = pyrender.Node(camera=self.camera, matrix=camera_pose)
            self.scene.add_node(nc)
            r = pyrender.OffscreenRenderer(800, 800)

            # Render an image
            color, depth = r.render(self.scene, flags=pyrender.constants.RenderFlags.RGBA)

            # Store the RGB image to the desired place
            image_rgb_name = dataset_location + image_dir + pose["image_name"]
            plt.imsave(image_rgb_name, color.copy(order='C'))


            # Convert the format and save depth image to the desired place
            depth = depth * 65535/(300)
            image_depth_name = dataset_location + image_dir + "depth_" + pose["image_name"]
            Image.fromarray(depth.astype('uint16')).save(image_depth_name)
            camera_pose_js = {}
            camera_pose_js["file_path"] = "." + image_dir + pose["image_name"]
            camera_pose_js["depth_path"] = "." + image_dir + "depth_" + pose["image_name"]
            # gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
            
            # camera_pose_js["sharpness"] = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Append the current image information to result_dict
            camera_pose_scale = camera_pose.tolist()
            camera_pose_scale[0][3] = camera_pose_scale[0][3]/64
            camera_pose_scale[1][3] = camera_pose_scale[1][3]/64
            camera_pose_scale[2][3] = camera_pose_scale[2][3]/64
            camera_pose_js["transform_matrix"] = camera_pose_scale
            self.result_dict["frames"].append(camera_pose_js)

            self.scene.remove_node(nc)
            r.delete()
        
        # Dump the data to json file
        json.dump(self.result_dict, camera_pose_file, indent = 4)
        camera_pose_file.close()
    def init_testbed(self, testbed_args):
        '''
        The function to initialize the testbed containing the NeRF model
        Input: configuration parameters stored in testbed_args
        '''
        #####################################################
        # initialize the testbed containing the NeRF model ##
        #####################################################
        if testbed_args.mode:
            print("Warning: the '--mode' argument is no longer in use. It has no effect. The mode is automatically chosen based on the scene.")

        self.testbed = ngp.Testbed()
        self.testbed.root_dir = ROOT_DIR

        # General purpose file loader (should be empty so far)
        for file in testbed_args.files:
            scene_info = self.get_scene(file)
            if scene_info:
                file = os.path.join(scene_info["data_dir"], scene_info["dataset"])
            self.testbed.load_file(file)

        # Load the training data
        if testbed_args.scene:
            # Load fundamental information about the scene (should be empty so far)
            scene_info = self.get_scene(testbed_args.scene)
            if scene_info is not None:
                testbed_args.scene = os.path.join(scene_info["data_dir"], scene_info["dataset"])
                if not testbed_args.network and "network" in scene_info:
                    testbed_args.network = scene_info["network"]
            
            # Load the training data (images so far)
            self.testbed.load_training_data(testbed_args.scene)


        # Whether to load a snapshot
        if testbed_args.load_snapshot:
            # Grab the basic information about the model snapshot, if applicable
            scene_info = self.get_scene(testbed_args.load_snapshot)
            if scene_info is not None:
                testbed_args.load_snapshot = default_snapshot_filename(scene_info)

            # Load the snapshot
            self.testbed.load_snapshot(testbed_args.load_snapshot)
        elif testbed_args.network:
            # Load the network parameters directly, if applicable
            self.testbed.reload_network_from_file(testbed_args.network)

        # Configure the poses to take screenshots on the NeRF scene
        self.ref_transforms = {}
        if testbed_args.screenshot_transforms: # try to load the given file straight away
            print("Screenshot transforms from ", testbed_args.screenshot_transforms)
            with open(testbed_args.screenshot_transforms) as f:
                self.ref_transforms = json.load(f)
        # Consider SDF case
        if self.testbed.mode == ngp.TestbedMode.Sdf:
            self.testbed.tonemap_curve = ngp.TonemapCurve.ACES

        # Additional configurations
        self.testbed.nerf.sharpen = float(testbed_args.sharpen)
        self.testbed.exposure = testbed_args.exposure

        # TODO: activate this boolean when training is needed
        self.testbed.shall_train = False 

        self.testbed.nerf.render_with_lens_distortion = True

        # Starting point of the model & More configurations on the model
        network_stem = os.path.splitext(os.path.basename(testbed_args.network))[0] if testbed_args.network else "base"
        if self.testbed.mode == ngp.TestbedMode.Sdf:
            setup_colored_sdf(self.testbed, testbed_args.scene)

        if testbed_args.near_distance >= 0.0:
            print("NeRF training ray near_distance ", testbed_args.near_distance)
            self.testbed.nerf.training.near_distance = testbed_args.near_distance

        if testbed_args.nerf_compatibility:
            print(f"NeRF compatibility mode enabled")

            # Prior nerf papers accumulate/blend in the sRGB
            # color space. This messes not only with background
            # alpha, but also with DOF effects and the likes.
            # We support this behavior, but we only enable it
            # for the case of synthetic nerf data where we need
            # to compare PSNR numbers to results of prior work.
            self.testbed.color_space = ngp.ColorSpace.SRGB

            # No exponential cone tracing. Slightly increases
            # quality at the cost of speed. This is done by
            # default on scenes with AABB 1 (like the synthetic
            # ones), but not on larger scenes. So force the
            # setting here.
            self.testbed.nerf.cone_angle_constant = 0

            # Match nerf paper behaviour and train on a fixed bg.
            self.testbed.nerf.training.random_bg_color = False

    def train_NeRF(self, testbed_args, start_from_prev_snapshot = False):
        ##########################################
        # Train the NeRF model & Evaluation ######
        ##########################################

        old_training_step = 0
        self.n_steps = testbed_args.n_steps
        self.testbed.shall_train = True
        # If we loaded a snapshot, didn't specify a number of steps, _and_ didn't open a GUI,
        # don't train by default and instead assume that the goal is to render screenshots,
        # compute PSNR, or render a video.

        training_steps = 2500
        if self.n_steps < 0 and (not testbed_args.load_snapshot or testbed_args.gui):
            self.n_steps = training_steps

        # Whether to activate gui
        if testbed_args.gui:
            # Pick a sensible GUI resolution depending on arguments.
            sw = testbed_args.width or 1920
            sh = testbed_args.height or 1080
            while sw * sh > 1920 * 1080 * 4:
                sw = int(sw / 2)
                sh = int(sh / 2)
            self.testbed.init_window(sw, sh, second_window=testbed_args.second_window)

        if start_from_prev_snapshot == True:
            # Load the previous stored snapshot
            self.testbed.load_snapshot(testbed_args.save_snapshot)
            self.n_steps = self.testbed.training_step + training_steps # Reset the training steps
        self.testbed.nerf.training.depth_supervision_lambda = testbed_args.depth_supervision_lambda
        tqdm_last_update = 0
        if self.n_steps > 0:
            with tqdm(desc="Training", total=self.n_steps, unit="steps") as t:
                while self.testbed.frame():
                    if self.testbed.want_repl():
                        repl(self.testbed)
                    # What will happen when training is done?
                    if self.testbed.training_step >= self.n_steps:
                        if testbed_args.gui:
                            self.testbed.shall_train = False
                        else:
                            break

                    # Update progress bar
                    if self.testbed.training_step < old_training_step or old_training_step == 0:
                        old_training_step = 0
                        t.reset()

                    now = time.monotonic()
                    if now - tqdm_last_update > 0.1:
                        t.update(self.testbed.training_step - old_training_step)
                        t.set_postfix(loss=self.testbed.loss)
                        old_training_step = self.testbed.training_step
                        tqdm_last_update = now
        # Save the snapshot on the model to the desired location
        if testbed_args.save_snapshot:
            os.makedirs(os.path.dirname(testbed_args.save_snapshot), exist_ok=True)
            self.testbed.save_snapshot(testbed_args.save_snapshot, False)

        ####################################################
        ## The followings are several tests after training ##
        ####################################################
        
        # Simple tests after a training - see the screenshots at several poses
        if testbed_args.test_transforms:
            print("Evaluating test transforms from ", testbed_args.test_transforms)

            # TODO: determine whether it's necessary to incorporate them here
            # (seems that they are not used)
            with open(testbed_args.test_transforms) as f:
                test_transforms = json.load(f)
            data_dir=os.path.dirname(testbed_args.test_transforms)

            totmse = 0
            totpsnr = 0
            totssim = 0
            totcount = 0
            minpsnr = 1000
            maxpsnr = 0

            # Evaluate metrics on black background
            self.testbed.background_color = [0.0, 0.0, 0.0, 1.0]

            # Prior nerf papers don't typically do multi-sample anti aliasing.
            # So snap all pixels to the pixel centers.
            self.testbed.snap_to_pixel_centers = True
            spp = 8

            self.testbed.nerf.render_min_transmittance = 1e-4

            self.testbed.shall_train = False
            self.testbed.load_training_data(testbed_args.test_transforms)

            with tqdm(range(self.testbed.nerf.training.dataset.n_images), unit="images", desc=f"Rendering test frame") as t:
                for i in t:
                    resolution = self.testbed.nerf.training.dataset.metadata[i].resolution
                    self.testbed.render_ground_truth = True
                    self.testbed.set_camera_to_training_view(i)
                    ref_image = self.testbed.render(resolution[0], resolution[1], 1, True)
                    self.testbed.render_ground_truth = False
                    image = self.testbed.render(resolution[0], resolution[1], spp, True)

                    if i == 0:
                        write_image(f"ref.png", ref_image)
                        write_image(f"out.png", image)

                        diffimg = np.absolute(image - ref_image)
                        diffimg[...,3:4] = 1.0
                        write_image("diff.png", diffimg)

                    A = np.clip(linear_to_srgb(image[...,:3]), 0.0, 1.0)
                    R = np.clip(linear_to_srgb(ref_image[...,:3]), 0.0, 1.0)
                    mse = float(compute_error("MSE", A, R))
                    ssim = float(compute_error("SSIM", A, R))
                    totssim += ssim
                    totmse += mse
                    psnr = mse2psnr(mse)
                    totpsnr += psnr
                    minpsnr = psnr if psnr<minpsnr else minpsnr
                    maxpsnr = psnr if psnr>maxpsnr else maxpsnr
                    totcount = totcount+1
                    t.set_postfix(psnr = totpsnr/(totcount or 1))

            psnr_avgmse = mse2psnr(totmse/(totcount or 1))
            psnr = totpsnr/(totcount or 1)
            ssim = totssim/(totcount or 1)
            print(f"PSNR={psnr} [min={minpsnr} max={maxpsnr}] SSIM={ssim}")

        # An extra test to save the mesh of the current 3D reconstructed scene
        if testbed_args.save_mesh:
            res = testbed_args.marching_cubes_res or 256
            thresh = testbed_args.marching_cubes_density_thresh or 2.5
            print(f"Generating mesh via marching cubes and saving to {testbed_args.save_mesh}. Resolution=[{res},{res},{res}], Density Threshold={thresh}")
            self.testbed.compute_and_save_marching_cubes_mesh(testbed_args.save_mesh, [res, res, res], thresh=thresh)

        # Save snapshots at several poses at the reference (similar to test_transform)
        if self.ref_transforms: # Already specified in the initialization
            self.testbed.fov_axis = 0
            self.testbed.fov = self.ref_transforms["camera_angle_x"] * 180 / np.pi
            if not testbed_args.screenshot_frames:
                testbed_args.screenshot_frames = range(len(self.ref_transforms["frames"]))
            print(testbed_args.screenshot_frames)
            for idx in testbed_args.screenshot_frames:
                f = self.ref_transforms["frames"][int(idx)]
                cam_matrix = f.get("transform_matrix", f["transform_matrix_start"])
                self.testbed.set_nerf_camera_matrix(np.matrix(cam_matrix)[:-1,:])
                outname = os.path.join(testbed_args.screenshot_dir, os.path.basename(f["file_path"]))

                # Some NeRF datasets lack the .png suffix in the dataset metadata
                if not os.path.splitext(outname)[1]:
                    outname = outname + ".png"

                print(f"rendering {outname}")
                image = self.testbed.render(testbed_args.width or int(self.ref_transforms["w"]), testbed_args.height or int(self.ref_transforms["h"]), testbed_args.screenshot_spp, True)
                os.makedirs(os.path.dirname(outname), exist_ok=True)
                write_image(outname, image)
        elif testbed_args.screenshot_dir:
            outname = os.path.join(testbed_args.screenshot_dir, testbed_args.scene + "_" + self.network_stem)
            print(f"Rendering {outname}.png")
            image = self.testbed.render(testbed_args.width or 1920, testbed_args.height or 1080, testbed_args.screenshot_spp, True)
            if os.path.dirname(outname) != "":
                os.makedirs(os.path.dirname(outname), exist_ok=True)
            write_image(outname + ".png", image)

        # Another test to capture a video
        if testbed_args.video_camera_path:
            self.testbed.load_camera_path(testbed_args.video_camera_path)

            resolution = [testbed_args.width or 1920, testbed_args.height or 1080]
            n_frames = testbed_args.video_n_seconds * testbed_args.video_fps
            save_frames = "%" in testbed_args.video_output
            start_frame, end_frame = testbed_args.video_render_range

            if "tmp" in os.listdir():
                shutil.rmtree("tmp")
            os.makedirs("tmp")

            for i in tqdm(list(range(min(n_frames, n_frames+1))), unit="frames", desc=f"Rendering video"):
                self.testbed.camera_smoothing = testbed_args.video_camera_smoothing

                if start_frame >= 0 and i < start_frame:
                    # For camera smoothing and motion blur to work, we cannot just start rendering
                    # from middle of the sequence. Instead we render a very small image and discard it
                    # for these initial frames.
                    # TODO Replace this with a no-op render method once it's available
                    frame = self.testbed.render(32, 32, 1, True, float(i)/n_frames, float(i + 1)/n_frames, testbed_args.video_fps, shutter_fraction=0.5)
                    continue
                elif end_frame >= 0 and i > end_frame:
                    continue

                frame = self.testbed.render(resolution[0], resolution[1], testbed_args.video_spp, True, float(i)/n_frames, float(i + 1)/n_frames, testbed_args.video_fps, shutter_fraction=0.5)
                if save_frames:
                    write_image(testbed_args.video_output % i, np.clip(frame * 2**testbed_args.exposure, 0.0, 1.0), quality=100)
                else:
                    write_image(f"tmp/{i:04d}.jpg", np.clip(frame * 2**testbed_args.exposure, 0.0, 1.0), quality=100)

            if not save_frames:
                os.system(f"ffmpeg -y -framerate {testbed_args.video_fps} -i tmp/%04d.jpg -c:v libx264 -pix_fmt yuv420p {testbed_args.video_output}")

            shutil.rmtree("tmp")




    def get_scene(self, scene):
        '''
        The function to get the scene
        '''
        for scenes in [scenes_sdf, scenes_nerf, scenes_image, scenes_volume]:
            if scene in scenes:
                return scenes[scene]
        return None
    
    def get_render_image(self, render_poses, mode = "entropy", dupl_allow = True, real_scene_scale = 64, snapshot_name = "/base.ingp", debug = True):
        '''
        The function to get the screenshots taken at the specified poses
            Entropy mode:
                The entropy snapshots will be the uncertainty estimator at that pose
            RGB mode:
                A visual image for reference
            Depth mode:
                A depth image for reference
        Input: render_poses: poses where the cameras are set up and take the snapshots (in real scene)
        Output: 
            Entropy mode:
                return the mean pixel entropy values at those poses
        '''
        #Loading Snapshot
        snapshot_dir = self.dataset_location + snapshot_name
        self.testbed.load_snapshot(snapshot_dir)

        resolution = [800, 800]
        spp = 8
        if debug:
            # Path to folder to save rendered photos in.
            folder = self.dataset_location + "/rendered_images_folder"

            # Intilazing list to save the transformation matrix of each rendered image.
            image_transformation = []

        # Load the base_cam json file
        
        ref_transforms = render_poses

        
        if (mode == "entropy"):
            self.testbed.render_mode = ngp.Entropy
            entropy_frame_mean = []
        elif (mode == "RGB"):
            self.testbed.render_mode = ngp.Shade
        elif (mode == "Depth"):
            self.testbed.render_mode = ngp.Depth
        else:
            print("Error! Unknown Rendering Mode")
            return
        
        # Looping over the path of each camera.
        for camera_pose in ref_transforms:
            # Field of view
            self.testbed.fov = camera_pose["fov"]

            
            cam_matrix = camera_pose["transformation"]
            if np.matrix(cam_matrix).shape[0] == 4:
                # If the matrix is given as 4x4
                cam_matrix = np.matrix(cam_matrix)[0:3]

            # The camera poses are given in the real scene
            # Should convert them into NeRF scene scale
            cam_matrix[0, 3] = cam_matrix[0, 3] / real_scene_scale
            cam_matrix[1, 3] = cam_matrix[1, 3] / real_scene_scale
            cam_matrix[2, 3] = cam_matrix[2, 3] / real_scene_scale
            # Revesring NGP axis, scaling and offset.
            #cam_matrix = ngp_to_nerf(cam_matrix)

            # Setting current transformation matrix.
            self.testbed.set_nerf_camera_matrix(np.matrix(cam_matrix))

            # Rendering Current image.
            if self.testbed.render_mode == ngp.Depth or self.testbed.render_mode == ngp.Entropy:
                frame = self.testbed.render(resolution[0],resolution[1],spp,linear=True)
                frame_copy = np.array(frame)

                frame_copy = np.ones((frame.shape[0], frame.shape[1]))
                frame_copy = frame[:, :, 0] # Extract out the linear units;

                if self.testbed.render_mode == ngp.Entropy:
                    # Two cases:
                    # If we allow choosing a selected camera pose, just append the mean value
                    # If not, for the already selected camera pose, append -1 
                    # (so that they will never be chosen)
                    if dupl_allow == False and camera_pose["selected"] == True:
                        entropy_frame_mean.append(-20.0)
                    else:
                        entropy_frame_mean.append(np.mean(frame_copy))
            else:
                frame = self.testbed.render(resolution[0],resolution[1],spp,linear=False)
                frame_copy = np.array(frame)
            # Store the image to the directory as a reference
            if debug: 
                # Current rendered image name (rendered_image_0.png,rendered_image_1.png,....)
                out_image_name = folder + "/" + mode + "_" + camera_pose["render_image_name"]

                # Saving the rendered image
                # common.write_image(out_image_name,frame,100)
                plt.imsave(out_image_name, frame_copy.copy(order='C'))

                # Stacking last row of the transformation matrix (4,4)
                cam_matrix = np.vstack((cam_matrix,[0,0,0,1]))
                
                # Apprending current rendered image transformation matrix to list.
                image_transformation.append({
                    "image_name": camera_pose["render_image_name"],
                    "transform_matrix": cam_matrix.tolist()
                })

                # Saving the rendered images Transformation matricies json file.
                image_transformation_path = os.path.join(folder, "rendered_images.json")
                
                with open(image_transformation_path, "w") as json_file:
                    json.dump(image_transformation, json_file, indent=4)
        if mode == "entropy":
            return entropy_frame_mean
        else:
            return
        

    def get_next_best_poses(self, camera_pose_candidates, dupl_allow = True, real_scene_scale = 64):
        '''
        The function to select the next best poses to take images from 
        According to the original paper, one per sector, and 12 in total
        Input:
            camera_pose_candidates:
                - All the candidates to take an image from; represent the view space of camera
            real_scene_scale:
                - the scale to match from nerf scene to the real scene (simulation/practical)
        Output:
            next_best_poses:
                - Several new poses to take an image at (12 in total currently)
        '''

        # Six sectors in the upper/lower half of the hemisphere
        sector_num = 12
        next_best_poses = []
        h_num = 5
        i_num = 30
        i_step = 5 # 30/6
        for sector in range(sector_num):
            # Which half the sector is at
            half = 0
            if sector >= 6:
                half = 3
            
            # Which sector it's in the current half
            idx = sector%6

            # if (sector == 1 or sector == 10):
            #     print("Test!!")
            #     print(sector)
            #     print(half * i_num + idx * i_step)
            # Concatenate the two parts in the sector
            camera_pose_sector_1 = camera_pose_candidates[
                (half) * i_num + idx * i_step : (half) * i_num + idx * i_step + i_step] 
            camera_pose_sector_2 = camera_pose_candidates[
                (half + 1) * i_num + idx * i_step : (half + 1) * i_num + idx * i_step + i_step]

            camera_pose_sector = camera_pose_sector_1 + camera_pose_sector_2

            entropy_values = self.get_render_image(
                    camera_pose_sector, 
                    mode = "entropy",  
                    dupl_allow = dupl_allow,
                    real_scene_scale=real_scene_scale,  
                    debug = False)
            idx = np.argmax(np.array(entropy_values)) # The image with the highest entropy/noise
            camera_pose_sector[idx]["selected"] = True
            next_best_poses.append(camera_pose_sector[idx])
        return next_best_poses
    

    