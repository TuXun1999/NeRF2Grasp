import open3d as o3d
import numpy as np
from mesh_process import *
from scipy.spatial.transform import Rotation as R
# Read the file as a triangular mesh
mesh = o3d.io.read_triangle_mesh("model_normalized.obj")
vis= o3d.visualization.Visualizer()
vis.create_window()

# Define the gripper
scale = 0.05
gripper_width = 2 * scale
gripper_length = 1 * scale
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

# Sample several points on the gripper
num_sample = 10
arm_end = np.array([gripper_length, 0, 0])
center = np.array([0, 0, 0])
elbow1 = np.array([0, 0, gripper_width/2])
elbow2 = np.array([0, 0, -gripper_width/2])
tip1 = np.array([-gripper_length, 0, gripper_width/2])
tip2 = np.array([-gripper_length, 0, -gripper_width/2])
gripper_part1 = np.linspace(arm_end, center, num_sample)
gripper_part2 = np.linspace(elbow1, tip1, num_sample)
gripper_part3 = np.linspace(elbow2, tip2, num_sample)
gripper_part4 = np.linspace(elbow1, elbow2, 2*num_sample)
gripper_points_sample = np.vstack((gripper_part1, gripper_part2, gripper_part3, gripper_part4))

# SE(3) Transformation on the gripper
transformation = np.eye(4)

quat = [0, np.sin(-np.pi/3), 0, np.cos(-np.pi/3)]
rot = R.from_quat(np.asarray(quat)).as_matrix()
transformation[0:3, 0:3] = rot
transformation[0:3, 3] = transformation[0:3, 3] - np.array([0.2, 0.12, -0.12])
grasp_pose = np.eye(4)
grasp_pose = np.matmul(transformation, grasp_pose)


grasp_pose_pc = o3d.geometry.PointCloud()
gripper_points_sample = np.vstack((gripper_points_sample.T, np.ones((1, gripper_points_sample.shape[0]))))
gripper_points_sample = (np.matmul(grasp_pose, gripper_points_sample)[0:3]).T

grasp_pose_pc.points = o3d.utility.Vector3dVector(gripper_points_sample)



gripper_attr = {"Length": gripper_length, "Width": gripper_width, "Thickness": gripper_length * 0.4}
# Test Collision
res, bbox = antipodal_test(mesh, grasp_pose, gripper_attr, 5, np.pi/36)
if res:
    grasp_pose_pc.paint_uniform_color((0, 1, 0))
else:
    grasp_pose_pc.paint_uniform_color((1, 0, 0))


# Plot out the fundamental frame
frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
frame.scale(4* scale, [0, 0, 0])

# TODO: debug the closing vector
vec_test = o3d.geometry.TriangleMesh.create_arrow()
vec_test.scale(0.1 * scale, [0, 0, 0])
vec_test.transform(grasp_pose)

vis.add_geometry(grasp_pose_pc)
vis.add_geometry(frame)
vis.add_geometry(mesh)
vis.add_geometry(bbox)
vis.add_geometry(vec_test)
vis.run()
vis.destroy_window()