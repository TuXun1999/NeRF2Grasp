import os, sys
import argparse
import common
pyngp_path = os.getcwd() + "/build"
sys.path.append(pyngp_path)
import pyngp as ngp
import numpy as np
import shutil
import argparse
import json
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
def ngp_to_nerf(cam_matrix):
   
    cam_matrix = cam_matrix[[2,0,1], :] # flip axis (yzx->xyz)
    cam_matrix[:,3] -= 0.5 # reversing offset
    cam_matrix[:,3] /= 0.33 # reversing scale
    cam_matrix[:,1] /= -1 # flipping y axis
    cam_matrix[:,2] /= -1 # z flipping
   
    return cam_matrix

def render_image(resolution,scene,spp):
    
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)

    #Loading Snapshot
    testbed.load_snapshot(os.path.join(scene,"base.ingp"))
    
    # Path to folder to save rendered photos in.
    folder = os.path.join(scene,"rendered_images_folder")
    
    # Delete the folder if it exists.
    if os.path.exists(folder) == True:
        shutil.rmtree(folder)

    # Intialize the folder.
    os.makedirs(folder)

    # Load the base_cam json file
    with open(os.path.join(scene,"base_cam.json")) as f:
        ref_transforms = json.load(f) 

    # Intilazing list to save the transformation matrix of each rendered image.
    image_transformation = []

    testbed.render_mode = ngp.Depth
    # Looping over the path of each camera.
    for i, current_path in enumerate(ref_transforms["path"]):
        
        # Field of view
        testbed.fov = current_path["fov"]

        # Transforming the rotation quaternion to a (3,3) Rotation matrix.
        cam_matrix = np.matrix([
                [
                    0.5000000000000007,
                    -0.5090369604551274,
                    0.7006292692220365,
                    1.3684165414492906
                ],
                [
                    0.8660254037844387,
                    0.293892626146237,
                    -0.40450849718747417,
                    -0.7900556585692854
                ],
                [
                    7.198293278059969e-17,
                    0.8090169943749477,
                    0.5877852522924736,
                    1.148018070883737
                ]
            ])

        # Revesring NGP axis, scaling and offset.
        #cam_matrix = ngp_to_nerf(cam_matrix)

        # Setting current transformation matrix.
        testbed.set_nerf_camera_matrix(np.matrix(cam_matrix))

        # Rendering Current image.
        frame = testbed.render(resolution[0],resolution[1],spp,linear=True)
        assert frame[0, 0, 0] == frame[0, 0, 1]
        print("Test!")
        print(frame[462][249])
        frame_copy = np.array(frame)
        
        if testbed.render_mode == ngp.Depth or testbed.render_mode == ngp.Entropy:
            
            frame_copy = frame[:, :, 0] # Extract out the linear units;
            plt.imshow(frame_copy, cmap=plt.cm.gray_r)
            plt.show()
        # Current rendered image name (rendered_image_0.png,rendered_image_1.png,....)
        out_image_name = os.path.join(folder, os.path.basename(f"rendered_image_{i}.png"))

        # Saving the rendered image
        # common.write_image(out_image_name,frame,100)
        plt.imsave(out_image_name, frame_copy.copy(order='C'))

        # Stacking last row of the transformation matrix (4,4)
        cam_matrix = np.vstack((cam_matrix,[0,0,0,1]))
        
        # Apprending current rendered image transformation matrix to list.
        image_transformation.append({
            "image_name": f"rendered_image_{i}.png",
            "transform_matrix": cam_matrix.tolist()
        })

    # Saving the rendered images Transformation matricies json file.
    image_transformation_path = os.path.join(folder, "rendered_images.json")
    
    with open(image_transformation_path, "w") as json_file:
        json.dump(image_transformation, json_file, indent=4)

    
 
def parse_args():
    parser = argparse.ArgumentParser(description="render neural graphics primitives testbed, see documentation for how to")
    parser.add_argument("--scene", "--training_data", default="", help="The path to the scene to load.")
    parser.add_argument("--width", "--screenshot_w", type=int, default=1920, help="Resolution width of the rendered image.")
    parser.add_argument("--height", "--screenshot_h", type=int, default=1080, help="Resolution height of the rendered image.")
    parser.add_argument("--spp", "--screenshot_spp", type=int, default=8, help="spp value.")


    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()	
    render_image([args.width, args.height], 
                 args.scene, 
                 args.spp)    