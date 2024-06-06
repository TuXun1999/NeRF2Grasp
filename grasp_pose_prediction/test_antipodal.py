import open3d as o3d
import numpy as np
from mesh_process import *
from scipy.spatial.transform import Rotation as R
# Read the file as a triangular mesh
mesh = o3d.io.read_triangle_mesh("./data/nerf/chair_sim_depth/chair_upper.obj")
vis= o3d.visualization.Visualizer()
vis.create_window()

# Define the gripper
scale = 10
gripper_width = 1 * scale
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
num_sample = 20
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


gripper_attr = {"Length": gripper_length, "Width": gripper_width, "Thickness": gripper_length * 0.4}
# TODO: debug vertices on the box
thickness = gripper_attr["Thickness"]
length = gripper_attr["Length"]
width = gripper_attr["Width"]

# Specify the box vertices
pt1 = np.array([0, thickness/2, width/2])
pt2 = np.array([-length, thickness/2, width/2])
pt3 = np.array([-length, -thickness/2, width/2])
pt4 = np.array([0, -thickness/2, width/2])

box_upper = np.vstack((pt1, pt2, pt3, pt4))
box_lower = copy.deepcopy(box_upper)
box_lower[:, 2] = -box_lower[:, 2]
box = np.vstack((box_upper, box_lower))


# SE(3) Transformation on the gripper
transformation = np.eye(4)

# quat = [np.sqrt(2)/2, np.sqrt(2)/2, np.sin(np.pi/3), np.cos(np.pi/3)]
# rot = R.from_quat(np.asarray(quat)).as_matrix()
# transformation[0:3, 0:3] = rot
# transformation[0:3, 3] = transformation[0:3, 3] - np.array([20, 20, 10])
transformation = np.array(
[[ 7.79390061e-03 , 8.61591416e-02 ,-9.96250901e-01,  1.83359385e+01],
 [ 9.87712259e-01 ,-1.56176473e-01, -5.77956784e-03,  3.00611938e+01],
 [-1.56088914e-01 ,-9.83964182e-01, -8.63176643e-02 , 2.90297886e+01],
 [ 0.00000000e+00, 0.00000000e+00,  0.00000000e+00 , 1.00000000e+00]])
grasp_pose = np.eye(4)
grasp_pose = np.matmul(transformation, grasp_pose)


grasp_pose_pc = o3d.geometry.PointCloud()
gripper_points_sample = np.vstack((gripper_points_sample.T, np.ones((1, gripper_points_sample.shape[0]))))
gripper_points_sample = (np.matmul(grasp_pose, gripper_points_sample)[0:3]).T

grasp_pose_pc.points = o3d.utility.Vector3dVector(gripper_points_sample)



# Test Collision
res, bbox, mesh_test= collision_test_local(mesh, gripper_points_sample, grasp_pose, gripper_attr, 0.05*gripper_width)
#res = collision_test(mesh, gripper_points, 0.05 * gripper_width)
if res:
    grasp_pose_pc.paint_uniform_color((1, 0, 0))
else:
    grasp_pose_pc.paint_uniform_color((0, 1, 0))
mesh_test.paint_uniform_color((1, 0, 0))

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
vis.add_geometry(mesh_test)
vis.run()
vis.destroy_window()