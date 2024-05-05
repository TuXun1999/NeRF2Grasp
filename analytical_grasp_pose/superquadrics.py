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
    angle_sample_space_num: the total number of angle candidates at the quarter
    sample_number: the expected number of grasp candidates sampled in the quarter
    tolerance: how far the gripper is away from the edge of the shape
    '''
    if e >= 1: # The easy case, where no "shrinking" occurs
        angle = np.linspace(0, np.pi/2, num=angle_sample_space_num)
        angle_sample = angle[np.random.randint(0, angle_sample_space_num, sample_number)]

        # Obtain a quarter of the final grasp candidates
        # The cos & sin values are always non-negative, because we only consider 0 - pi/2
        res_x = (a1 + tolerance) * np.power(np.cos(angle_sample), e)
        res_y = (a2 + tolerance) * np.power(np.sin(angle_sample), e)

        # Convert the results into columns
        res_x = np.asarray(res_x).reshape(-1, 1)
        res_y = np.asarray(res_y).reshape(-1, 1)

        # Force the grippers to look into the origin
        angle_sample = np.arctan2(res_y, res_x).reshape(-1, 1)
        
    else:
        # TODO: figure out the situation where e<1, and there will be inconsistent parts
        # Use straight lines to fit the "empty" parts

        angle = np.linspace(0, np.pi/2, num=angle_sample_space_num)
        angle_sample = angle[np.random.randint(0, angle_sample_space_num, sample_number)]
        x_der = e * a1 * np.power(np.cos(angle), e - 1) * np.sin(angle)
        y_der = e * a2 * np.power(np.sin(angle), e - 1) * np.cos(angle)

        # Find the angle values where x_der, y_der boost up 
        angle_critical_x_idx = np.argmin(abs(x_der - 4 * e * a1)) # The critical angle where dx boosts up
        angle_critical_x = angle[angle_critical_x_idx]
        angle_critical_y_idx  = np.argmin(abs(y_der - 4 * e * a2)) # The critical angle where dy boosts up
        angle_critical_y = angle[angle_critical_y_idx]

        # The critical points
        # The critical point for the right uncontinuous part
        right_uncont_x0 = a1 * np.power(np.cos(angle_critical_y), e)
        right_uncont_y0 = a2 * np.power(np.sin(angle_critical_y), e)

         # Find the critical point for the top uncontinuous part
        top_uncont_x0 = a1 * np.power(np.cos(angle_critical_x), e)
        top_uncont_y0 = a2 * np.power(np.sin(angle_critical_x), e)


        # Sample grasp candidates in the continuous region at first
        right_uncont_angle = np.arctan2(right_uncont_y0, right_uncont_x0)
        top_uncont_angle = np.arctan2(top_uncont_y0, top_uncont_x0)

        # According to the ratio of the continuous region,
        # determine the number of grasp candidates in that part
        cont_ratio = (top_uncont_angle - right_uncont_angle) / (np.pi/2)
        angle_sample_cont_num = (int)(cont_ratio * sample_number)

        # Only sample a few candidates from the continuous part
        angle_sample_cont = angle[\
            np.random.randint(angle_critical_y_idx, \
                              angle_critical_x_idx, \
                                angle_sample_cont_num)]
        res_x_cont = (a1 + tolerance) * np.power(np.cos(angle_sample_cont), e)
        res_y_cont = (a2 + tolerance) * np.power(np.sin(angle_sample_cont), e)

        # Sample grasp candidates in the un-continuous regions
        uncont_cand_space = 20

        # Find the number of candidates from the right un-continuous part
        right_sample_num = (int)((sample_number - angle_sample_cont_num)/ 2)
        right_theta = np.arctan2(a1 - right_uncont_x0, right_uncont_y0)

        # The candidate space for the right uncontinuous part
        right_uncont_xcoord = np.linspace(right_uncont_x0, a1, uncont_cand_space)
        
        right_uncont_xcoord = right_uncont_xcoord[\
            np.random.randint(0, uncont_cand_space - 1, right_sample_num)]
        # Use geometric relationship to sample several candidates
        right_uncont_ycoord = (right_uncont_y0 / (a1 - right_uncont_x0)) * (a1 - right_uncont_xcoord)
        res_x_right_uncont = right_uncont_xcoord + tolerance * np.cos(right_theta)
        res_y_right_uncont = right_uncont_ycoord + tolerance * np.sin(right_theta)

        angle_sample_right_uncont = np.ones(right_sample_num) * (right_theta)
       
        top_sample_num = sample_number - angle_sample_cont_num - right_sample_num
        top_theta = np.arctan2(a2 - top_uncont_y0, top_uncont_x0)

        # The candidate space for the top uncontinuous part
        top_uncont_ycoord = np.linspace(top_uncont_y0, a2, uncont_cand_space)
        top_uncont_ycoord = top_uncont_ycoord[\
            np.random.randint(0, uncont_cand_space - 1, top_sample_num)]
        
        # Use geometric properties to determine the locations of the points
        top_uncont_xcoord = (top_uncont_x0 / (a2 - top_uncont_y0)) * (a2 - top_uncont_ycoord)
        res_y_top_uncont = top_uncont_ycoord + tolerance * np.cos(top_theta)
        res_x_top_uncont = top_uncont_xcoord + tolerance * np.sin(top_theta)
        
        angle_sample_top_uncont = np.ones(top_sample_num) * (np.pi/2 - top_theta)
        # Stack all results together
        res_x = np.hstack((res_x_right_uncont, res_x_cont, res_x_top_uncont))
        res_y = np.hstack((res_y_right_uncont, res_y_cont, res_y_top_uncont))
        angle_sample = np.hstack((angle_sample_right_uncont, \
                                  angle_sample_cont, angle_sample_top_uncont))
        
        # Reshape the results into columns
        res_x = res_x.reshape(-1, 1)
        res_y = res_y.reshape(-1, 1)
        angle_sample = angle_sample.reshape(-1, 1)
    

   

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
    angle_sample_space_num = 50
    
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
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
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
    
epsilon1 = 0.1
epsilon2 = 1.9
a1 = 1.5
a2 = 0.4
a3 = 1.2

pc = create_superellipsoids(epsilon1, epsilon2, a1, a2, a3)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc)

# Visualize the super-ellipsoids
pcd.colors = o3d.utility.Vector3dVector(np.ones(pc.shape).astype(np.float64) / 255)

gripper_width = 2.0
gripper_length = 1.0
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

