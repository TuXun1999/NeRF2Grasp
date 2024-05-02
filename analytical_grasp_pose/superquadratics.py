import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

import math
import scipy
from scipy.spatial.transform import Rotation as R
import open3d as o3d

def create_superellipsoids(e1, e2, a1, a2, a3):
    pc_shape = 50
    pc = np.zeros((pc_shape * pc_shape, 3))
    idx = 0
    for i in np.linspace(-np.pi/2, np.pi/2, pc_shape):
        for j in np.linspace(-np.pi, np.pi, pc_shape, endpoint=False):
            eta =  i
            omega = j

            t1 = np.sign(np.cos(eta)) * np.power(abs(np.cos(eta)), e1)
            t2 = np.sign(np.sin(eta)) * np.power(abs(np.sin(eta)), e1)
            t3 = np.sign(np.cos(omega)) * np.power(abs(np.cos(omega)), e2)
            t4 = np.sign(np.sin(omega)) * np.power(abs(np.sin(omega)), e2)
            pc[idx, 0] = a1 * t1 * t3
            pc[idx, 1] = a2 * t1 * t4
            pc[idx, 2] = a3 * t2

            idx = idx + 1
    return pc

def grasp_pose_sample_quarter(a1, a2, e, angle_sample_space_num, sample_number, tolerance, reflection=0):
    '''
    The function to sample several grasp poses at the specified sampled angles
    Input: reflection: the direction to reflect the sampled grasp poses
    '''
    if e >= 1: # The easy case, where no "shrinking" occurs
        angle = np.linspace(0, np.pi/2, num=angle_sample_space_num)
        angle_sample = angle[np.random.randint(0, angle_sample_space_num, sample_number)]
        # Obtain a quarter of the final grasp candidates
        res_x = (a1 + tolerance) * np.sign(np.cos(angle_sample)) * np.power(abs(np.cos(angle_sample)), e)
        res_y = (a2 + tolerance) * np.sign(np.sin(angle_sample)) * np.power(abs(np.sin(angle_sample)), e)
        
        res_x = np.asarray(res_x).reshape(-1, 1)
        res_y = np.asarray(res_y).reshape(-1, 1)
        angle_sample = angle_sample.reshape(-1, 1)
    else:
        # TODO: figure out the situation where e<1, and there will be inconsistent parts
        pass

    # Do the necessary reflections
    if reflection == 0: # Quarter I
        res = np.hstack((res_x, res_y, angle_sample))
    elif reflection == 1: # Quarter II
        res = np.hstack((-res_x, res_y, np.pi - angle_sample))
    elif reflection == 2: # Quarter III
        res = np.hstack((-res_x, -res_y, np.pi + angle_sample))
    elif reflection == 3: # Quarter IV
        res = np.hstack((res_x, -res_y, 2 * np.pi - angle_sample))
    else:
        print("Unknown reflection direction")
        assert False
    assert res.shape[1] == 3
    return res
def grasp_pose_predict_sq(a1, a2, e, sample_number = 20, tolerance=0.5):
    '''
    The function to predict several grasp poses for a superquadric on xy-plane
    Only for grasp poses on the plane
    Input: a1, a2: the lengths of the two axes
            e: the power of the sinusoidal terms
    '''
    angle_sample_space_num = 100
    
    # Reflect the results to the other regions
    res_part1 = grasp_pose_sample_quarter(a1, a2, e, angle_sample_space_num, \
                                          (int)(sample_number/4), tolerance, reflection=0)
    res_part2 = grasp_pose_sample_quarter(a1, a2, e, angle_sample_space_num, \
                                        (int)(sample_number/4), tolerance, reflection=1)
    res_part3 = grasp_pose_sample_quarter(a1, a2, e, angle_sample_space_num, \
                                        (int)(sample_number/4), tolerance, reflection=2)
    res_part4 = grasp_pose_sample_quarter(a1, a2, e, angle_sample_space_num, \
                                        (int)(sample_number/4), tolerance, reflection=3)

    res = np.vstack((res_part1, res_part2, res_part3, res_part4))
    return res

def transform_matrix_convert(grasp_poses, principal_axis):
    '''
    The function to convert the grasp poses from grasp pose prediction (sq) 
    to standard 4x4 transformation matrices
    '''
    result = []
    for i in range(grasp_poses.shape[0]):
        x, y, angle = grasp_poses[i]
        rot = R.from_quat([0, 0, np.sin(angle/2), np.cos(angle/2)]).as_matrix()
        tran = np.vstack((np.hstack((rot, np.array([x,y,0]).reshape(-1,1))), np.array([0, 0, 0, 1])))

        if principal_axis == 2: # z is the shortest axis in length
            pass
        elif principal_axis == 1: # y is the shortest axis in length
            tran = np.matmul(np.array([
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1]
            ]), tran)
        elif principal_axis == 0: # x is the shortest axis in length
            tran = np.matmul(np.array([
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ]), tran)
        result.append(tran)
    return result
    
epsilon1 = 1.2
epsilon2 = 1.5
a1 = 0.8
a2 = 1.0
a3 = 1.6

pc = create_superellipsoids(epsilon1, epsilon2, a1, a2, a3)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc)

# Visualize the super-ellipsoids
pcd.colors = o3d.utility.Vector3dVector(np.ones(pc.shape).astype(np.float64) / 255)

gripper_width = 1.3
gripper_length = 0.8
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
    [0.8, 0, 0],
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
for grasp_pose in grasp_poses:
    grasp_pose_lineset = o3d.geometry.LineSet()

    gripper_points_vis = np.vstack((gripper_points.T, np.ones((1, gripper_points.shape[0]))))
    gripper_points_vis = np.matmul(grasp_pose, gripper_points_vis)
    grasp_pose_lineset.points = o3d.utility.Vector3dVector(gripper_points_vis[:-1].T)
    grasp_pose_lineset.lines = o3d.utility.Vector2iVector(gripper_lines)
    grasp_pose_lineset.colors = o3d.utility.Vector3dVector(gripper_colors)
    
    vis.add_geometry(grasp_pose_lineset)

# Plot out the fundamental frame
frame_points = [
    [0, 0, 0],
    [0.5, 0, 0],
    [0, 0.5, 0],
    [0, 0, 0.5]
]
frame_lines = [
    [0, 1],
    [0, 2],
    [0, 3]
]
frame_colors = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
]


frame = o3d.geometry.LineSet()
frame.points = o3d.utility.Vector3dVector(frame_points)
frame.lines = o3d.utility.Vector2iVector(frame_lines)
frame.colors = o3d.utility.Vector3dVector(frame_colors)



vis.add_geometry(pcd)
vis.add_geometry(frame)





vis.run()
vis.destroy_window()

