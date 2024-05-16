import json
import numpy as np
import cv2
import copy
from PIL import Image
from scipy.optimize import fsolve
# mouse callback function
g_image_click = None
g_image_display = None
def cv_mouse_callback(event, x, y, flags, param):
    global g_image_click, g_image_display
    clone = copy.deepcopy(g_image_display)
    if event == cv2.EVENT_LBUTTONUP:
        g_image_click = (x, y)
    elif not (g_image_click is None):
        # Draw some lines on the image.
        # to indicate the location of the selected point
        color = (30, 30, 30)
        thickness = 2
        image_title = 'Click'
        height = clone.shape[0]
        width = clone.shape[1]
        cv2.line(clone, (0, g_image_click[1]), (width, g_image_click[1]), color, thickness)
        cv2.line(clone, (g_image_click[0], 0), (g_image_click[0], height), color, thickness)
        cv2.circle(clone, g_image_click, radius = 4, color=color)
        cv2.imshow(image_title, clone)
    else:
        # Draw some lines on the imaege.
        #print('mouse', x, y)
        color = (30, 30, 30)
        thickness = 2
        image_title = 'Click'
        height = clone.shape[0]
        width = clone.shape[1]
        cv2.line(clone, (0, y), (width, y), color, thickness)
        cv2.line(clone, (x, 0), (x, height), color, thickness)
        cv2.imshow(image_title, clone)

def get_pick_vec_manual_force(img):
    global g_image_display, g_image_click
    g_image_display = img

    # Show the image to the user and wait for them to click on a pixel
    image_title = 'Click'
    print(image_title)
    cv2.namedWindow(image_title, cv2.WINDOW_AUTOSIZE)
    print("Set up callback")
    cv2.setMouseCallback(image_title, cv_mouse_callback)


    print("Show the image")
    cv2.imshow(image_title, g_image_display)
    while g_image_click is None:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            # The user decides not to pick anything in the frame
            return 0.0, 0.0
    return g_image_click[0], g_image_click[1]
def read_proj_from_json(dataset_dir, img_name, nerf_scale=64):
    # Read the json file
    json_filename = dataset_dir + "/transforms.json"
    f = open(json_filename)
    data = json.load(f)

    fl_x = data["fl_x"]
    fl_y = data["fl_y"]
    cx = data["cx"]
    cy = data["cy"]

    proj = np.array([
        [fl_x, 0, cx, 0],
        [0, fl_y, cy, 0],
        [0, 0, 1, 0]
    ])
    # By default, it's assumed that the camera is at the origin
    camera_pose = np.eye(4)
    # Search for the frame with the name matched
    for frame in data["frames"]:
        if frame["file_path"] == "." + img_name: 
            # When recording the name of the image, "." is always included
            camera_pose = frame["transform_matrix"]
            camera_pose = np.array(camera_pose)
            camera_pose[0:3, 3] = camera_pose[0:3, 3] * nerf_scale
            break
    # Conversion between the camera frame in simple-pinhole model & 
    # the actual transformation applied to the camera itself
    camera_pose = np.matmul(\
        camera_pose, \
            np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0 , 1]]))

    return proj, camera_pose


def point_select_from_image(img_dir, img_file, nerf_scale):
    ## Part I: open an image
    # Open the image (human choice)
    image = cv2.imread(img_dir + img_file)

    # Select a point manually
    pick_x, pick_y = get_pick_vec_manual_force(image)

    print("Clicked Point")
    print(pick_x)
    print(pick_y)


    ## Part II: obtain the corresponding camera projection matrix
    camera_proj, camera_pose = read_proj_from_json(img_dir, img_file, nerf_scale = nerf_scale)
    print("Camera pose in world frame: ")
    print(camera_pose)


    ## Part III: solve out the depth ambiguity (skipped)

    ## Part IV: find the ray direction in camera frame
    initial_guess = [1, 1, 10]
    def equations(vars):
        x, y, z = vars
        eq = [
            camera_proj[0][0] * x + camera_proj[0][1] * y + camera_proj[0][2] * z - pick_x * z,
            camera_proj[1][0] * x + camera_proj[1][1] * y + camera_proj[1][2] * z - pick_y * z,
            x * x + y * y + z * z - 10 * 10
        ]
        return eq

    root = fsolve(equations, initial_guess)

    
    ## Part V: conver the point coorindate in world frame
    dir_world = np.matmul(camera_pose, np.array([[root[0]], [root[1]], [root[2]], [0]]))
    return dir_world[0:3, 0], camera_pose, camera_proj[:, :-1]