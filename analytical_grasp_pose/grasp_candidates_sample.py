import numpy as np
import open3d as o3d
from gripper import gripper_V_shape
from darboux_frame import fit_neighbor, calculate_darboux_frame
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import KDTree
from point_cloud_process import point_in_gripper
import point_cloud_utils as pcu
## The main file to sample several grasp poses on the point cloud
## as well as evaulate the grasp qualities


# Build the V-shape gripper
gripper = gripper_V_shape(0.5, 0.3, 0.1, 0.1, 0.1, 0.5, scale=0.4)
gripper.open_gripper(np.pi/2)

# Read the npy file
pc_file = np.load("point_cloud_fuze.npy", allow_pickle=True)
pc_v, pc_f = pcu.load_mesh_vf("fuze.ply")
print(pc_f)
pc_file = pc_file.item()
pc = pc_file['xyz']
pc_colors = pc_file['xyz_color']
pc_tree = KDTree(pc)
print(len(pc))
print('Visualizing...')


# Visualize the data
vis = o3d.visualization.Visualizer()
pc_num = len(pc)
p_sel_idx = 1862 #np.random.randint(pc_num) #1080 #1570 #142
p_sel = pc[p_sel_idx]
print(p_sel_idx)



# Visualize the calculated Darboux frame
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc)
pcd.colors = o3d.utility.Vector3dVector(pc_colors.astype(np.float64) / 255)

c = fit_neighbor(pc_tree, pc, p_sel, pcd)
df_axis_1, df_axis_2, p_sel_N_curvature = calculate_darboux_frame(c, p_sel, vis)
print(df_axis_1)
print(df_axis_2)
print(p_sel_N_curvature)


## Visualize the gripper

## Apply a transformation to fit the gripper to the darboux frame
darboux_frame_rot = np.transpose(np.vstack((df_axis_1, df_axis_2, p_sel_N_curvature)))
darboux_frame_tran = np.array([
    [p_sel[0]],
    [p_sel[1]],
    [p_sel[2]]
])
# Convert it into a transformation matrix
darboux_frame = np.vstack((np.hstack((darboux_frame_rot, darboux_frame_tran)), np.array([0, 0, 0, 1])))

# Local changes in the darboux frame
# Initially, the gripper is placed along the x-axis
# Apply these changes:
# 1. Rotate around y-axis (for -pi/2)
# 2. Elevate the gripper for 0.5
df_tran1 = R.from_quat([0, np.sin(np.pi/4), 0, np.cos(np.pi/4)]).as_matrix()
df_tran1 = np.vstack((np.hstack((df_tran1, np.array([[0],[0],[0]]))), np.array([0, 0, 0, 1])))
df_tran2 = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0.3],
    [0, 0, 0, 1]
])
df_tran_local = np.matmul(df_tran2, df_tran1)
df_tran = np.matmul(darboux_frame, df_tran_local)
gripper.apply_transformation(df_tran)

grasp_candidates = []

# Find a large region around the sample point for collision check
check_collision_pc_idx = pc_tree.query_radius(p_sel.reshape(1, -1), r=0.15)

# Do a grid search to generate several grasp candidates
for x in range(4):
    # Move the gripper a step forward (stepsize: 0.1)
    tran_d = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, -0.05],
            [0, 0, 0, 1]
        ])
    df_cand_translate = np.matmul(darboux_frame, np.matmul(tran_d, np.linalg.inv(darboux_frame)))
    gripper.apply_transformation(df_cand_translate)

    go_to_next_dist = False
    for rot in range(6):
        if go_to_next_dist == True:
            # If we have found a good grasp pose at this distance, go to next distance
            continue
        # Rotate the gripper for an angle
        rot_angle = np.pi/12
        tran_rot = R.from_quat([0, 0, np.sin(rot_angle), np.cos(rot_angle)]).as_matrix()
        tran_rot = np.vstack((np.hstack((tran_rot, np.array([[0],[0],[0]]))), np.array([0, 0, 0, 1])))
        df_cand_rotate = np.matmul(darboux_frame, np.matmul(tran_rot, np.linalg.inv(darboux_frame)))
        gripper.apply_transformation(df_cand_rotate)


        collision = False
        print("Calculate Min dist...")
        for part in gripper.parts:
            # Test the collision between each part of the gripper and the collision region

            dists, fid, bc = pcu.closest_points_on_mesh(part, pc_v, pc_f)
            collision_dist = np.min(dists)
            
            if collision_dist < 0.01: #Too close
                collision = True
                print("Collision Detected !")
                break 
            else:
                continue
        if collision: # If there is a collision, go to next candidate
            continue
        else: # If not, check whether the sampled point is within the gripper's grasping region
            print("Point in Gripper?")
            # If the point is outside the grasping region,
            # it won't fall into the region again, even if you rotate it

            # If the point is inside the grasping region,
            # we have already obtained a good grasp pose, and no need to rotate it either
            go_to_next_dist = True
            
            in_gripper, quality = point_in_gripper(p_sel.flatten(), gripper=gripper)
            if in_gripper:
                # If the point is within the grasping region, push it to the grasp candidates
                grasp_candidate_origin = gripper.frame[0:3, 3] # Origin of the gripper
                grasp_candidate_point_upper = gripper.upper_gripper[25*51-1] # Mid-point of the upper gripper
                grasp_candidate_point_lower = gripper.lower_gripper[25*51-1] # Mid-point of the lower gripper
                grasp_candidate = np.vstack((
                    grasp_candidate_origin,
                    grasp_candidate_point_upper,
                    grasp_candidate_point_lower,
                    np.array([quality, quality, quality])
                ))
                print("New grasp candidate")
                grasp_candidates.append(grasp_candidate)
            else:
                continue


vis.create_window()
vis.add_geometry(pcd)
df_axis_1, df_axis_2, p_sel_N_curvature = calculate_darboux_frame(c, p_sel, vis, verbose = True)

# Visualize the grasp candidates
for candidate in grasp_candidates:
    p1 = candidate[0, :]
    p2 = candidate[1, :]
    p3 = candidate[2, :]
    color = candidate[3, 0]
    grasp_candidate_points = [
            [p1[0], p1[1], p1[2]],
            [p2[0], p2[1], p2[2]],
            [p3[0], p3[1], p3[2]]
        ]
    grasp_candidate_lines = [
        [0, 1],
        [0, 2]
    ]
    grasp_candidate_colors = [
        [1 * color, 1- color, 0],
        [1 * color, 1 - color, 0]
    ]


    grasp_candidate_V = o3d.geometry.LineSet()
    grasp_candidate_V.points = o3d.utility.Vector3dVector(grasp_candidate_points)
    grasp_candidate_V.lines = o3d.utility.Vector2iVector(grasp_candidate_lines)
    grasp_candidate_V.colors = o3d.utility.Vector3dVector(grasp_candidate_colors)
    vis.add_geometry(grasp_candidate_V)

# gripper_pcd = []
# for part in gripper.parts:
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(part)
#     pcd.colors = o3d.utility.Vector3dVector(0.1 * np.ones(part.shape))
#     vis.add_geometry(pcd)
    
vis.run()
vis.destroy_window()