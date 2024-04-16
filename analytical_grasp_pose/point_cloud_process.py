import numpy as np
import math
from scipy.spatial.transform import Rotation as R
def min_dist(pc1, pc1_idx, pc2, pc2_idx):
    '''
    Calculate the point pair with the minimum distance between pc1 and pc2
    Input: 
    pc1, pc2 - Two point cloud Data. Shape: Nx3
    Output:
    p1, p2: The two points in the point pair with the minimum distance;
    where p1 comes from pc1, p2 comes from pc2
    dist: The minimum distance
    Description: Suppose pc1, pc2 are O(n), then the cost is O(n log n), instead of O(n^2)
    '''
    l1 = pc1_idx
    l2 = pc2_idx
    min_dist = 10000
    for i in l1:
        for j in l2:
            p1 = pc1[i]
            p2 = pc2[j]
            dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)
            if  dist < min_dist:
                min_dist = dist
    return min_dist

def overlap(pc1, pc2, th):
    '''
    Check whether the two point cloud collides with each other
    '''
    p1, p2, dist = min_dist(pc1, pc2)
    if dist < th:
        return True, p1, p2
    else:
        return False, p1, p2

def point_in_gripper(p, gripper):
    '''
    The function to check whether the point is within the closing region of the gripper
    Input:
    p: the sampled point, a list of three elements
    gripper: the model of gripper used in the experiments
    '''
    p_homo = np.array([
        [p[0]],
        [p[1]],
        [p[2]],
        [1]
    ])
    gripper_origin = gripper.frame[:, 3].reshape(-1, 1)

    local_vec = np.matmul(np.linalg.inv(gripper.frame), p_homo  - gripper_origin)

    # Check whether the point satisfies the two conditions
    gw = gripper.attributes["gripper_width"]
    gl = gripper.attributes["gripper_length"]
    y_range = (local_vec[1][0] < gw/2) and (local_vec[1][0] > -gw/2)


    angle = math.atan2(local_vec[2][0],local_vec[0][0])
    angle_range = (angle > -gripper.open_angle/2) and (angle < gripper.open_angle/2)

    dist = math.sqrt(local_vec[2][0] * local_vec[2][0] + local_vec[0][0] * local_vec[0][0])
    dist_range = (dist < 0.9 * gl)
    
    in_gripper = y_range and angle_range and dist_range
    return in_gripper, dist_range/gl


## Section II: functions to determine whether the given point cloud 
## forms a handle

def points_proj_to_plane(pc_sel, pc_neighbor, df_axis_1, df_axis_2, p_sel_N_curvature):
    # Project the local neighboring region around the selected point
    # onto a plane orthogonal to the direction of minimum curvature
    # Input: pc_neighbor: Nx3
    # Output: pr_proj: 2xN, points in the orthogonal plane
    
    # Construct the transformation frame
    transformation = np.hstack((df_axis_1.reshape(-1,1), df_axis_2.reshape(-1,1), \
                                p_sel_N_curvature.reshape(-1,1), pc_sel.reshape(-1,1)))
    transformation = np.vstack((transformation, np.array([0, 0, 0, 1])))

    # Build up the matrix of all points
    N = pc_neighbor.shape[0]
    pc_neighbor_homo = np.transpose(pc_neighbor)
    pc_neighbor_homo = np.vstack((pc_neighbor_homo, np.ones((1, N))))

    # Find the projected points in the local frame & Extract the y&z coordinates
    pc_proj = np.matmul(np.linalg.inv(transformation), pc_neighbor_homo)[1:3, :]
    return pc_proj

def fit_cylinder_shell(pc_proj):
    # After project the points onto the plane, check whether
    # the points can be fit to a cylinder shell
    # Input: pc_proj: 2xN
    # Return:
    # hx, hy: center of the circle in the LOCAL frame
    # r: radius of the circle
    # error: average fitting error
    N = pc_proj.shape[1]
    L = -np.vstack((pc_proj, -np.ones((1, N))))
    # Calculate \sum_i l_i l_i^T
    LLi = np.matmul(L, np.transpose(L))

    # Calculate \lambda, where \lambda_i = x_i^2 + y_i^2
    lambda_i = np.sum(pc_proj[0:2] * pc_proj[0:2], axis=0).reshape(-1, 1)
    L_lambda = np.matmul(L, lambda_i)

    # Calculate the coefficients according to the formulas given in the paper
    w = -np.matmul(np.linalg.inv(LLi), L_lambda).flatten()

    hx = 0.5 * w[0]
    hy = 0.5 * w[1]
    r = math.sqrt(hx*hx + hy*hy - w[2])

    circle_center = np.array([[hx], [hy]])
    error_matrix = pc_proj - circle_center
    error_matrx = error_matrix * error_matrix
    error_vector = np.sum(error_matrix, axis=0) - r*r
    error = np.mean(error_vector)

    return hx, hy, r, error

def generate_contact_graspnet_file(gripper_pose, pc_sel_neighbor):
    '''
    The function to generate the pcd file to be fed into 
    Contact GraspNet to obtain the grasp candidates
    Input: gripper_pose: 4x4 transformation matrix, the transformation of the gripper frame
        pc_sel_neighbor: Nx3, the neighboring region of the selected point 
    Output: 
        a .npy file for the Contact GraspNet
    '''
    # Convert the local region into the standard format
    pc_sel_neighbor_homo = np.vstack((pc_sel_neighbor.T, np.ones((1, pc_sel_neighbor.shape[0]))))
    local_points = np.matmul(np.linalg.inv(gripper_pose), pc_sel_neighbor_homo)

    # Obey the frame convention of Contact GraspNet (z -> from the camera to the object)
    local_transformation = R.from_quat([0, np.sin(-np.pi/4), 0, np.cos(-np.pi/4)]).as_matrix()
    local_transformation = np.vstack(\
        (np.hstack((local_transformation, np.array([[0],[0],[0]]))), np.array([0, 0, 0, 1])))
    local_points = np.matmul(local_transformation, local_points)

    # (Optional) Test Equivariance, by rotating the points along z at a random angle
    angle = 0 #2 * math.pi * np.random.random()
    rotZ = R.from_quat([0, 0, np.sin(angle/2), np.cos(angle/2)]).as_matrix()
    rotZ = np.vstack(\
        (np.hstack((rotZ, np.array([[0],[0],[0]]))), np.array([0, 0, 0, 1])))
    local_points = np.matmul(rotZ, local_points)
    # Write up the npy file
    c = np.ones((local_points.shape[1], 3))/ 255
    npy_data = {}
    npy_data['xyz_color'] = c
    npy_data['xyz'] = local_points[0:3, :].T
    camera_intrinsic = np.array([
        [400, 0, 400],
        [0, 400, 400],
        [0, 0, 1]
    ])
    npy_data['K'] = camera_intrinsic

    np.save("../../contact_graspnet_pytorch/test_data/chair_local_region", npy_data)
