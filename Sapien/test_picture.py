"""Camera.

Concepts:
    - Create and mount cameras
    - Render RGB images, point clouds, segmentation masks
"""

import sapien.core as sapien
import numpy as np
from PIL import Image, ImageColor
import open3d as o3d
from sapien.utils.viewer import Viewer
from transforms3d.euler import mat2euler

import numpy as np
import os
import shutil
from scipy.spatial.transform import Rotation as R
import json


def main():
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    scene = engine.create_scene()
    scene.set_timestep(1 / 100.0)
    ray_tracing = True
    if ray_tracing:
        sapien.render_config.camera_shader_dir = "rt"
        sapien.render_config.viewer_shader_dir = "rt"
        # sapien.render_config.rt_samples_per_pixel = 64
        # sapien.render_config.rt_use_denoiser = True
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    urdf_path = './test_data/chair1/mobility.urdf'
    # load as a kinematic articulation
    asset = loader.load_kinematic(urdf_path)
    assert asset, 'URDF not loaded.'
    scene.add_ground(-0.9)


    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    # ---------------------------------------------------------------------------- #
    # Camera
    # ---------------------------------------------------------------------------- #
    near, far = 0.1, 100
    width, height = 640, 480
    camera = scene.add_camera(
        name="camera",
        width=width,
        height=height,
        fovy=np.deg2rad(90),
        near=near,
        far=far,
    )
    
    print('Intrinsic matrix\n', camera.get_intrinsic_matrix())
    
    # Configure the json file
    camera_pose_file = open("./transforms.json", "w")
    # Obtain the intrinsic parameters of the camera
    nerf_scale = 0.5 # The scale to fit the object into the unit cube

    result_dict = {}
    result_dict["w"] = 640
    result_dict["h"] = 480
    result_dict["aabb_scale"] = 1
    result_dict["fl_x"] = 240
    result_dict["k1"] = 0
    result_dict["p1"] = 0
    result_dict["fl_y"] = 240
    result_dict["k2"] = 0
    result_dict["p2"] = 0
    result_dict["cx"] = 320
    result_dict["cy"] = 240
    result_dict["camera_angle_x"] = np.pi/2
    result_dict["camera_angle_y"] = np.pi/2
    result_dict["enable_depth_loading"] = True
    result_dict["integer_depth_scale"] = 2/(nerf_scale * 65535)
    result_dict["frames"] = []

    folder = "./images"
    # Delete the folder if it exists.
    if os.path.exists(folder) == True:
        shutil.rmtree(folder)

    # Intialize the folder.
    os.makedirs(folder)


    # Initial rotations to adjust the pose of the camera

    # Sapien Camera: initially, the camera faces +x direction, +y as the lateral
    # Rotate around the z-axis for pi
    rot = R.from_quat([0, 0, np.sin(np.pi/2), np.cos(np.pi/2)]).as_matrix()

    # Translation part
    d = np.array([[1.5], [0], [0]])

    # Transformation matrix at the initial pose
    trans_initial = np.vstack((np.hstack((rot, d)), np.array([0, 0, 0, 1])))

    # NeRF/Pyrender Camera: initially, the camera faces +z direction, +x as the lateral
    # Rotate around the x-axis for pi/2
    rot1_nerf = R.from_quat([np.sin(np.pi/4), 0, 0, np.cos(np.pi/4)]).as_matrix()

    # Rotate around the z-axis for pi/2
    rot2_nerf= R.from_quat([0, 0, np.sin(-np.pi/4), np.cos(-np.pi/4)]).as_matrix()
    
    rot_nerf = np.matmul(rot2_nerf, rot1_nerf)
    rot_nerf = np.vstack((np.hstack((rot_nerf, np.array([[0], [0], [0]]))), np.array([0, 0, 0, 1])))
    # Number of samples vertically
    h_num = 5
    # Number of samples along the circle around the object
    i_num = 30

    angle_dev_i = 2*np.pi/i_num
    angle_dev_h = (np.pi/2)/h_num
   

    h_numbers = range(h_num)
    i_numbers = range(i_num)
    for h in h_numbers: # Rotate around axis-y
        rot_angle_h = -h * angle_dev_h
        # Rotation matrix for the camera pose around y-axis
        rot_around_x = R.from_quat([0, np.sin(rot_angle_h/2), 0, np.cos(rot_angle_h/2)]).as_matrix()
        camera_pose_vet = np.matmul(
                np.vstack(
                    (np.hstack((rot_around_x, np.array([[0], [0], [0]]))), 
                    np.array([0, 0, 0, 1]))),
                trans_initial
                )
        for i in i_numbers: # Rotate around axis-z
            rot_angle_i  = i * angle_dev_i

            # Rotation matrix for that camera pose
            rot_around_z = R.from_quat([0, 0, np.sin(rot_angle_i/2), np.cos(rot_angle_i/2)]).as_matrix()
            # Move camera to that sampling pose
            camera_pose = np.matmul(
                np.vstack(
                    (np.hstack((rot_around_z, np.array([[0], [0], [0]]))), 
                    np.array([0, 0, 0, 1]))),
                camera_pose_vet
                )

            r = list(R.from_matrix(camera_pose[0:3, 0:3]).as_quat())
            if i == 0 and h == 0:
                print(r)
                
            camera.set_pose(sapien.Pose(p=list(camera_pose[0:3, 3]), q=[r[3], r[0], r[1], r[2]]))
            
            scene.step()  # make everything set
            scene.update_render()
            camera.take_picture()
            # ---------------------------------------------------------------------------- #
            # RGBA
            # ---------------------------------------------------------------------------- #
            rgba = camera.get_float_texture('Color')  # [H, W, 4]
            # An alias is also provided
            # rgba = camera.get_color_rgba()  # [H, W, 4]
            rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
            
            # Filter out the background (in simulation)
            hi, wi, _ = rgba_img.shape
            for hii in range(hi):
                 for wii in range(wi):
                    bg = rgba_img[hii, wii]
                    if ray_tracing == True:
                        if bg[0] == 186 and bg[1] == 186 and bg[2] == 186 and bg[3] == 255:
                            # Value of the background from experiments (should be compatible)
                            rgba_img[hii, wii] = np.array([0, 0, 0, 0])
                    else:
                        if bg[0] == 0 and bg[1] == 0 and bg[2] == 0 and bg[3] == 255:
                            # Value of the background from experiments (should be compatible)
                            rgba_img[hii, wii] = np.array([0, 0, 0, 0])

            rgba_pil = Image.fromarray(rgba_img, 'RGBA')
            rgba_pil.save("./images/chair_" + str(h) + "_" + str(i) + ".png")

            # ----------------------------------------------------------------------------#
            # Depth
            # ----------------------------------------------------------------------------#
            # ---------------------------------------------------------------------------- #
            # XYZ position in the camera space
            # ---------------------------------------------------------------------------- #
            # Each pixel is (x, y, z, render_depth) in camera space (OpenGL/Blender)
            position = camera.get_float_texture('Position')  # [H, W, 4]

        
           
            # Depth
            depth = -position[..., 2]
            depth_image = (depth * 65535/2).astype(np.uint16)
            Image.fromarray(depth_image).save("./images/chair_depth_" + str(h) + "_" + str(i) + ".png")
            camera_pose_js = {}
            camera_pose_js["file_path"] = "./images/chair_" + str(h) + "_" + str(i) + ".png"
            camera_pose_js["depth_path"] = "./images/chair_depth_" + str(h) + "_" + str(i) + ".png"
            # gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
            
            # camera_pose_js["sharpness"] = cv2.Laplacian(gray, cv2.CV_64F).var()
            camera_pose_scale = (np.matmul(camera_pose, rot_nerf)).tolist()
            camera_pose_scale[0][3] = camera_pose_scale[0][3]/nerf_scale
            camera_pose_scale[1][3] = camera_pose_scale[1][3]/nerf_scale
            camera_pose_scale[2][3] = camera_pose_scale[2][3]/nerf_scale
            camera_pose_js["transform_matrix"] = camera_pose_scale
            result_dict["frames"].append(camera_pose_js)


    # Dump the data to json file
    json.dump(result_dict, camera_pose_file, indent = 4)
    camera_pose_file.close()


    

   


    # ---------------------------------------------------------------------------- #
    # Take picture from the viewer
    # ---------------------------------------------------------------------------- #
    viewer = Viewer(renderer)
    viewer.set_scene(scene)
    # We show how to set the viewer according to the pose of a camera
    # opengl camera -> sapien world
    model_matrix = camera.get_model_matrix()
    # sapien camera -> sapien world
    # You can also infer it from the camera pose
    model_matrix = model_matrix[:, [2, 0, 1, 3]] * np.array([-1, -1, 1, 1])
    # The rotation of the viewer camera is represented as [roll(x), pitch(-y), yaw(-z)]
    rpy = mat2euler(model_matrix[:3, :3]) * np.array([1, -1, -1])
    viewer.set_camera_xyz(*model_matrix[0:3, 3])
    viewer.set_camera_rpy(*rpy)
    viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)
    while not viewer.closed:
        if viewer.window.key_down('p'):  # Press 'p' to take the screenshot
            rgba = viewer.window.get_float_texture('Color')
            rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
            rgba_pil = Image.fromarray(rgba_img)
            rgba_pil.save('screenshot.png')
        scene.step()
        scene.update_render()
        viewer.render()


if __name__ == '__main__':
    main()
