import cv2
import numpy as np
from scipy.optimize import fsolve
import copy
import open3d as o3d
import json
from PIL import Image
from superquadrics import *
from preprocess import *
import os
from scipy.spatial.transform import Rotation as R
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
        # Draw some lines on the image.
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
    return dir_world[0:3, 0], camera_pose


    

if __name__ == "__main__":
    nerf_scale = 64

    ## The image used to specify the selected point
    img_dir = "../data/nerf/chair_sim_depth"
    img_file = "/images/chair_2_20.png"

    ## Obtain the ray direction of the selected point in space 
    dir, camera_pose = point_select_from_image(img_dir, img_file, nerf_scale=nerf_scale)
    dir = dir / np.linalg.norm(dir)

    ## Determine the selected point location in 3D space
    # Specify the ply file
    filename="chair_upper.obj"
    
    # Read the file as a triangular mesh
    mesh = o3d.io.read_triangle_mesh(filename)

    normalize_obj = "model_normalized.obj"
    if not os.path.isfile(normalize_obj): # Correct the coordinate convention if the normalized model is not generated
        # Restore the correct scale & fix up the coordinate issue 
        mesh = coordinate_correction(mesh, filename, nerf_scale)

        # Output the normalized triangular mesh & Obtain the stats
        stats = model_normalized(filename, normalize_obj, \
                                 normalize_stats_file="normalize.npy", stats = None)
    else:
        # If the normalize model already exists, read the parameters directly
        stats = read_normalize_stats("normalize.npy").item()
    
    # TODO: transfer the codes here to generate the predicted superquadrics file


    # Create a scene & Add the mesh to the scene
    scene = o3d.t.geometry.RaycastingScene()
    mesh_id = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))


    # Cast the ray from the camera
    camera_location = camera_pose[0:3, 3]
    ray = list(np.hstack((camera_location, dir)))
    rays = o3d.core.Tensor([ray],
                       dtype=o3d.core.Dtype.Float32)

    ans = scene.cast_rays(rays)

    # Obtain the distance upon hitting
    point_select_distance = min(ans['t_hit'].numpy()[0], 200)

    # Calculate the 3D point coordinates
    pos = camera_pose[0:3, 3] + dir * point_select_distance
    print(point_select_distance)
    print(dir)
    print('Visualizing...')

    # TODO: Crop a local region at the specified point


    ## Display the selected point
    # Create the window to pick up the desired point
    vis= o3d.visualization.Visualizer()
    vis.create_window()


    # Add a sphere to the selected point
    ball_select =  o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    ball_select.scale(1/64 * nerf_scale, [0, 0, 0])

    ball_select.translate((pos[0], pos[1], pos[2]))
    ball_select.paint_uniform_color((1, 0, 0))

    # Plot out the camera frame
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    camera_frame.scale(20/64 * nerf_scale, [0, 0, 0])
    camera_frame.transform(camera_pose)
    

    ## Read the parameters of the superquadrics and import the point
    sq_dir = "./test_tmp"
    sq_vertices = []
    sq_transformation = []

    # Extract out the necessary attributes
    diag = stats['max'] - stats['min']
    norm = np.linalg.norm(diag)
    c = stats['centroid']
    # Read the sq attributes in each file
    for sq_file in os.listdir(sq_dir):
        parameters = read_sq_parameters(sq_dir + "/" + sq_file)
        if parameters is None:
            continue 
        prob = parameters["probability"][0]
        if prob < 0.5:
            continue

        # Read the parameters
        epsilon1 = parameters["shape"][0]
        epsilon2 = parameters["shape"][1]
        a1 = parameters["size"][0]
        a2 = parameters["size"][1]
        a3 = parameters["size"][2]

        # Sample the points on the superquadric
        pc = create_superellipsoids(epsilon1, epsilon2, a1, a2, a3)

        # Apply the transformation
        quat = parameters["rotation"]
        quat = np.array(quat)
        quat[[0, 1, 2, 3]] = quat[[1, 2, 3, 0]]


        translation = np.asarray(parameters["location"])
        
        rot = R.from_quat(quat).as_matrix()

        # Obtain the correct point coordinates in the normalized frame
        pc_tran = rot.T.dot(pc.T) + translation.reshape(3, -1)
        pc_tran = pc_tran.T
        sq_tran = np.vstack((np.hstack((rot.T, translation.reshape(3, -1))), np.array([0, 0, 0, 1])))

        # Revert the normalization process
        pc_tran = pc_tran * norm + c
        sq_tran[0:3, 3] = sq_tran[0:3, 3] * norm + c

        sq_vertices.append(pc_tran)
        sq_transformation.append({"sq_parameters": parameters, \
                                    "points": pc_tran, \
                                    "transformation": sq_tran})
    
    
    # Construct a point cloud representing the reconstructed object mesh
    pcd = o3d.geometry.PointCloud()
    sq_vertices = np.array(sq_vertices).reshape(-1, 3)
    
    # Displacement observed from experiments...
    displacement = -np.mean(sq_vertices, axis=0) + np.mean(np.asarray(mesh.vertices), axis=0)
    pcd.points = o3d.utility.Vector3dVector(sq_vertices + displacement)
    # Visualize the super-ellipsoids
    pcd_colors = np.repeat(np.array([[0.0, 255.0, 0.0]])/255, sq_vertices.shape[0], axis=0)
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

    ## Find the sq associated to the selected point
    # Evaluate the transformation of each sq
    dist_min = np.Infinity
    sq_closest = None
    for sq_tran_dict in sq_transformation:
        # Read the transformation of this sq & several sample points on it
        sq_tran = sq_tran_dict["transformation"]
        pc_tran = sq_tran_dict["points"]
        parameters = sq_tran_dict["sq_parameters"]
        
        # Apply the displacement
        sq_tran[0:3, 3] = sq_tran[0:3, 3] + displacement
        pc_tran = pc_tran + displacement

        

        # Read the parameters of this sq
        epsilon1 = parameters["shape"][0]
        epsilon2 = parameters["shape"][1]
        a1 = parameters["size"][0] * norm
        a2 = parameters["size"][1] * norm
        a3 = parameters["size"][2] * norm
        
        # Find the point's coordinate in the sq's frame
        pos_sq = np.matmul(np.linalg.inv(sq_tran), np.array([\
                    [pos[0]], [pos[1]], [pos[2]], [1]]\
                ))

        x,y,z,_ = pos_sq.flatten()
        
        # Calculate the evaluation value
        x1 = x/a1
        y1 = y/a2
        z1 = z/a3

        # Calculate the evaluation value F(x0, y0, z0)
        val1 = np.power(x1*x1, 1/epsilon2) + np.power(y1*y1, 1/epsilon2)
        val2 = np.power(val1, epsilon2/epsilon1) + np.power(z1*z1, 2/epsilon1)
        # beta calculation
        beta = np.power(val2, -epsilon1/2)

        dist = abs(1 - beta) * np.sqrt(x**2 + y**2 + z**2)

        # Find the closest sq to the selected point
        if dist < dist_min :
            dist_min = dist
            sq_closest = {}
            sq_closest["sq_parameters"] = parameters
            sq_closest["transformation"] = sq_tran
            sq_closest["points"] = pc_tran

            

    # Color the associated sq in blue
    pcd_associated = o3d.geometry.PointCloud()
    pcd_associated.points = o3d.utility.Vector3dVector(sq_closest["points"])
    pcd_associated.colors = o3d.utility.Vector3dVector(\
                    np.repeat(np.array([[0.0, 0.0, 255.0]])/255, pc_tran.shape[0], axis=0)
                )
    
    # Plot out the fundamental frame of the associated sq
    sq_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    sq_frame.scale(20/64 * nerf_scale, [0, 0, 0])
    sq_frame.transform(sq_closest["transformation"])
    print("Final test")
    print(sq_closest["sq_parameters"])
    print(norm)
    vis.add_geometry(sq_frame)

    ## Determine the grasp candidates on this sq and visualize them
    # Read the parameters of the associated sq
    parameters_closest = sq_closest["sq_parameters"]
    epsilon1_closest = parameters_closest["shape"][0]
    epsilon2_closest = parameters_closest["shape"][1]
    a1_closest = parameters_closest["size"][0] * norm
    a2_closest = parameters_closest["size"][1] * norm
    a3_closest = parameters_closest["size"][2] * norm
    
    # Find the principal axis & Call the function to determine the grasp poses
    scale = 10
    gripper_width = 2 * scale
    gripper_length = 1 * scale

    # Sample a series of grasp poses
    min_idx = np.argmin(np.array([a1_closest, a2_closest, a3_closest]))
    principal_axis = 2
    if min_idx == 2: # If z is the direction of the the shortest axis in length
        grasp_poses = grasp_pose_predict_sq(a1_closest, a2_closest, epsilon2_closest, tolerance= gripper_length/2)
    elif min_idx == 1: # If y is the direction of the shortest axis in length
        grasp_poses = grasp_pose_predict_sq(a1_closest, a3_closest, epsilon1_closest, tolerance=gripper_length/2)
        principal_axis = 1
    else: # If x is the direction of the shorest axis in length
        grasp_poses = grasp_pose_predict_sq(a2_closest, a3_closest, epsilon1_closest, tolerance=gripper_length/2)
        principal_axis = 0
    # Transform the grasp poses to the correct positions in world frame
    grasp_poses = transform_matrix_convert(grasp_poses, principal_axis)

     # Construct the gripper
    gripper_points = np.array([
        [0, 0, 0],
        [gripper_length, 0, 0],
        [0, 0, gripper_width/2],
        [0, 0, -gripper_width/2],
        [-gripper_length, 0, gripper_width/2],
        [-gripper_length, 0, -gripper_width/2]
    ])
    gripper_lines = [
        [1, 0],
        [2, 3],
        [2, 4],
        [3, 5]
    ]
    gripper_colors = [
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0]
    ]
    # Construct the grasp poses at the specified locations,
    # and add them to the visualizer
    for grasp_pose in grasp_poses:
        # Find the grasp pose in the world frame (converted from sq local frame)
        grasp_pose = np.matmul(sq_closest["transformation"], grasp_pose)
        grasp_pose_lineset = o3d.geometry.LineSet()

        gripper_points_vis = np.vstack((gripper_points.T, np.ones((1, gripper_points.shape[0]))))
        gripper_points_vis = np.matmul(grasp_pose, gripper_points_vis)
        grasp_pose_lineset.points = o3d.utility.Vector3dVector(gripper_points_vis[:-1].T)
        grasp_pose_lineset.lines = o3d.utility.Vector2iVector(gripper_lines)
        grasp_pose_lineset.colors = o3d.utility.Vector3dVector(gripper_colors)
        
        vis.add_geometry(grasp_pose_lineset)
    ## Postlogue
    # Plot out the fundamental frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    frame.scale(20/64 * nerf_scale, [0, 0, 0])


    vis.add_geometry(mesh)
    vis.add_geometry(pcd)
    vis.add_geometry(pcd_associated) 
    vis.add_geometry(frame)
    vis.add_geometry(camera_frame)
    vis.add_geometry(ball_select)
    vis.run()
    vis.destroy_window()


