import numpy as np
import math

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