import numpy as np
import os
import sys
sys.path.append(os.getcwd())

import open3d as o3d
from superquadrics import create_superellipsoids, grasp_pose_predict_sq, read_sq_parameters, transform_matrix_convert
import scipy
from scipy.spatial.transform import Rotation as R




# parameters = read_sq_parameters("./test_tmp/primitive_0.p") 
# epsilon1 = parameters["shape"][0]
# epsilon2 = parameters["shape"][1]
# a1 = parameters["size"][0] * 64
# a2 = parameters["size"][1] * 64
# a3 = parameters["size"][2] * 64

norm = 122.8209
x,y,z = 14.79, 0.7060, -17.01035

scale = 0.1
epsilon1 = 0.1
epsilon2 = 0.1
a1 = 0.1232 * norm
a2 = 0.13 * norm
a3 = 0.8 * norm
x1 = x/a1
y1 = y/a2
z1 = z/a3
print(a1)
print(a2)
print(a3)
# Calculate the evaluation value F(x0, y0, z0)
val1 = np.power(x1*x1, 1/epsilon2) + np.power(y1*y1, 1/epsilon2)
val2 = np.power(val1, epsilon2/epsilon1) + np.power(z1*z1, 2/epsilon1)
# beta calculation
beta = np.power(val2, -epsilon1/2)
print("Test")
print(x1)
print(y1)
print(z1)
print(val1)
print(val2)
print(beta)
pc = create_superellipsoids(epsilon1, epsilon2, a1, a2, a3)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc)

# Visualize the super-ellipsoids
pcd.colors = o3d.utility.Vector3dVector(np.ones(pc.shape).astype(np.float64) / 255)
scale = 10
gripper_width = 2 * scale
gripper_length = 1 * scale
# Use the methodology to sample a series of grasp poses
min_idx = np.argmin(np.array([a1, a2, a3]))
principal_axis = 2
if min_idx == 2: # If z is the direction of the the shortest axis in length
    grasp_poses = grasp_pose_predict_sq(a1, a2, epsilon2, tolerance= gripper_length/2)
elif min_idx == 1: # If y is the direction of the shortest axis in length
    grasp_poses = grasp_pose_predict_sq(a1, a3, epsilon1, tolerance=gripper_length/2)
    principal_axis = 1
else: # If x is the direction of the shorest axis in length
    grasp_poses = grasp_pose_predict_sq(a2, a3, epsilon1, tolerance=gripper_length/2)
    principal_axis = 0

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
# Visualization 
vis = o3d.visualization.Visualizer()
vis.create_window()


grasp_poses = transform_matrix_convert(grasp_poses, principal_axis)

# Construct the grasp poses at the specified locations,
# and add them to the visualizer
# for grasp_pose in grasp_poses:
#     grasp_pose_lineset = o3d.geometry.LineSet()

#     gripper_points_vis = np.vstack((gripper_points.T, np.ones((1, gripper_points.shape[0]))))
#     gripper_points_vis = np.matmul(grasp_pose, gripper_points_vis)
#     grasp_pose_lineset.points = o3d.utility.Vector3dVector(gripper_points_vis[:-1].T)
#     grasp_pose_lineset.lines = o3d.utility.Vector2iVector(gripper_lines)
#     grasp_pose_lineset.colors = o3d.utility.Vector3dVector(gripper_colors)
    
#     vis.add_geometry(grasp_pose_lineset)

# Plot out the fundamental frame
frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
frame.scale(scale, [0, 0, 0])



vis.add_geometry(pcd)
vis.add_geometry(frame)





vis.run()
vis.destroy_window()

