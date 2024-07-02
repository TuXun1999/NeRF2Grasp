import numpy as np
import sys
import os
sys.path.append(os.getcwd())
import open3d as o3d
import csv
from Marching_Primitives.MPS import MPS, eul2rotm, parseInputArgs
from superquadrics import create_superellipsoids, nms_sq_bbox
import scipy.io


# Loading file paths
csvfile_path = "./data/nerf/chair7_pm/chair_upper.csv" # Specify the location of the csv file
csvfile_path_list = csvfile_path.split('.')

# Verify that the selected file is in csv format
if csvfile_path_list[-1] != "csv":
    print("Please select a csv file!!")

mesh_filename = csvfile_path
normalize_suffix_idx = mesh_filename.find("_normalized.csv")
# Check whether the model is normalized
if normalize_suffix_idx != -1:
    mesh_filename = mesh_filename[0:normalize_suffix_idx] + ".obj"
else:
    csv_suffix_idx = mesh_filename.find(".csv")
    mesh_filename = mesh_filename[0:csv_suffix_idx] + ".obj"


# Read the csv file & Extract out SDF
sdf = []
with open(csvfile_path, newline='') as csvfile:
    sdf_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in sdf_reader:
        sdf.append(float(row[0]))
sdf = np.array(sdf)
print(sdf[0])

# Build up the voxel grid
voxelGrid = {}
voxelGrid['size'] = (np.ones(3) * sdf[0]).astype(int)
voxelGrid['range'] = sdf[1:7]
sdf = sdf[7:]

voxelGrid['x'] = np.linspace(float(voxelGrid['range'][0]), float(voxelGrid['range'][1]), int(voxelGrid['size'][0]))
voxelGrid['y'] = np.linspace(float(voxelGrid['range'][2]), float(voxelGrid['range'][3]), int(voxelGrid['size'][1]))
voxelGrid['z'] = np.linspace(float(voxelGrid['range'][4]), float(voxelGrid['range'][5]), int(voxelGrid['size'][2]))
x, y, z = np.meshgrid(voxelGrid['x'], voxelGrid['y'], voxelGrid['z'])

# Permute the order (different data orders in Matlab & Python)
# NOTE: This part is just trying to obey the data order convention in original 
# Matlab program. There might be a way to continue the program even in Python data order
x = np.transpose(x, (1, 0, 2))
y = np.transpose(y, (1, 0, 2))
z = np.transpose(z, (1, 0, 2))

# Construct the points in voxelGrid 
s = np.stack([x, y, z], axis=3)
s = s.reshape(-1, 3, order='F').T

# Construct the voxel grid
voxelGrid['points'] = s

voxelGrid['interval'] = (voxelGrid['range'][1] - voxelGrid['range'][0]) / (voxelGrid['size'][0] - 1)
voxelGrid['truncation'] = 1.2 * voxelGrid['interval']
voxelGrid['disp_range'] = [-np.inf, voxelGrid['truncation']]
voxelGrid['visualizeArclength'] = 0.01 * np.sqrt(voxelGrid['range'][1] - voxelGrid['range'][0])

# Complte extracting out the sdf
sdf = np.clip(sdf, -voxelGrid['truncation'], voxelGrid['truncation'])

# Visualizing SDF values
sdf_pc = o3d.geometry.PointCloud()
sdf_pc.points = o3d.utility.Vector3dVector(s.T)
sdf_max = np.max(sdf)
sdf_min = np.min(sdf)
print(sdf.shape)
color = ((sdf - sdf_min) / (sdf_max - sdf_min)).reshape(-1, 1)
print(color.shape)
colr = color * np.array([-1, 0, 1]) 
print(color.shape)
color = color + np.array([1, 0, 0])
print(color.shape)
sdf_pc.colors = o3d.utility.Vector3dVector(color)
# marching-primitives
import time

# Parsing varargin
para = parseInputArgs(voxelGrid, sys.argv[1:])
start_time = time.time()
# x = MPS(sdf, voxelGrid, para) 
# print(x[0])
# This line is to read results from Matlab programs (mainly for debugging purpose)
x = scipy.io.loadmat('./grasp_pose_prediction/matlab_res.mat').get('x')
x, bbox_test, iou_3d = nms_sq_bbox(x, 0.4)
print(f"Elapsed time: {time.time() - start_time:.2f} seconds")


# Read the original mesh file (open3d is used here for visualization)
# TODO: could use other libraries, such as trimehs, maya, etc. 
mesh = o3d.io.read_triangle_mesh(mesh_filename)
if normalize_suffix_idx != -1:
    # Normalize the original mesh (for visualization purpose)
    mesh_scale = 0.8
    vertices = np.asarray(mesh.vertices)
    bbmin = np.min(vertices, axis=0)
    bbmax = np.max(vertices, axis=0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

sq_mesh = []
sq_center = []
sq_bbox = []
for i in range(x.shape[0]):
    # Read the superquadrics' parameters
    e1, e2, a1, a2, a3, r, p, y, t1, t2, t3 = x[i, :]
    if e1 < 0.01:
        e1 = 0.01
    if e2 < 0.01:
        e2 = 0.01

    # Custom function to sample points on the superquadrics
    sq_vertices = create_superellipsoids(e1, e2, a1, a2, a3)
    rot = eul2rotm(np.array([r, p, y]))
    sq_vertices = np.matmul(rot, sq_vertices.T).T + np.array([t1, t2, t3])

    # Construct a point cloud representing the reconstructed object mesh
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sq_vertices)
    # Visualize the super-ellipsoids
    pcd.paint_uniform_color((0.0, 0.4, 0.4))

    sq_mesh.append(pcd)

    # Visualize the center of the superquadrics
    ball_select =  o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    ball_select.scale(1/64, [0, 0, 0])

    ball_select.translate((t1, t2, t3))
    ball_select.paint_uniform_color((0.4, 0.4, 0.4))
    sq_center.append(ball_select)

    # Visualize the bounding box
    # Points on the bounding box of the sq
    pc = np.array([
        [-a1, -a2, -a3],
        [a1, -a2, -a3],
        [a1, a2, -a3],
        [-a1, a2, -a3],
        [-a1, -a2, a3],
        [a1, -a2, a3],
        [a1, a2, a3],
        [-a1, a2, a3]
    ])

    # Create the bounding box at the initial position (same original as the world frame)
    bbox = o3d.geometry.OrientedBoundingBox().\
        create_from_points(o3d.utility.Vector3dVector(pc))
    bbox.rotate(rot)
    bbox.translate(np.array([t1, t2, t3]))
    bbox.color = (0, 0.4, 0.4)
    sq_bbox.append(bbox)

# Try to draw lines between connected SQ (somehow...)
k = np.nonzero(np.array(iou_3d))
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(x[:, -3:])
lines = np.vstack((k[0], k[1])).T

line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.colors = o3d.utility.Vector3dVector([[0, 0.4, 0] for i in range(x.shape[0])])

# Construct a convex hull on the centers of the sq's
pcl= o3d.geometry.PointCloud()
pcl.points = o3d.utility.Vector3dVector(x[:, -3:])

hull, hull_indices = pcl.compute_convex_hull()
hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
hull_ls.paint_uniform_color((1, 0, 0))

# Emphasize the vertices of the convex hull
hull_v_mesh = []
hull_v = np.array(hull_ls.points)
for i in hull_v:
    hull_v_ball = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    hull_v_ball.scale(1/64, [0, 0, 0])

    hull_v_ball.translate((i[0], i[1], i[2]))
    hull_v_ball.paint_uniform_color((0, 0.4, 0))
    hull_v_mesh.append(hull_v_ball)


# Find out and Plot the camera frame
nerf_scale = 1
camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
camera_frame.scale(20/64 * nerf_scale, [0, 0, 0])
camera_pose = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, -1],
    [0, 0, 1, 0.8],
    [0, 0, 0, 1]
])
camera_frame.transform(camera_pose)

camera_t = camera_pose[0:3, 3]
idx = np.argmin(np.linalg.norm(hull_v - camera_t, axis=1))
hull_v_mesh[idx].paint_uniform_color((1, 0, 0))

# Create the window to display everything
vis= o3d.visualization.Visualizer()
vis.create_window()
#vis.add_geometry(mesh)
for val in sq_mesh:
    vis.add_geometry(val)
for val in sq_center:
    vis.add_geometry(val)
# for val in sq_bbox:
#     vis.add_geometry(val)
for val in hull_v_mesh:
    vis.add_geometry(val)


# Construct a point cloud representing the reconstructed object mesh
pcd_bbox = o3d.geometry.PointCloud()
pcd_bbox.points = o3d.utility.Vector3dVector(np.array(bbox_test).reshape(-1, 3))
# Visualize the super-ellipsoids
pcd_bbox.paint_uniform_color((0.4, 0.4, 0.0))
# vis.add_geometry(pcd_bbox)
# vis.add_geometry(line_set)
# vis.add_geometry(hull_ls)

vis.add_geometry(camera_frame)

vis.add_geometry(sdf_pc)
vis.run()

# Close all windows
vis.destroy_window()
