import numpy as np
import open3d as o3d
from gripper import gripper_V_shape
from darboux_frame import fit_neighbor, calculate_darboux_frame
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import KDTree
from point_cloud_process import point_in_gripper, generate_contact_graspnet_file
import point_cloud_utils as pcu
## The main file to sample several grasp poses on the point cloud
## as well as evaulate the grasp qualities

nerf_scale = 64/100


## The main file to sample several grasp poses on the point cloud
## as well as evaulate the grasp qualities

nerf_scale = 0.64

'''
Section I: Read the whole object
'''
# Read the ply file
filename="chair2.ply"
file1 = open(filename, "r")

# pc_v, pc_f are the original complete file
pc_v, pc_f = pcu.load_mesh_vf(filename)
if pc_f is None:
    pc_f = []
    # Using readlines()
    lines = file1.readlines()
    
    # Strips the newline character
    for line in lines:
        line = line.split()
        if line[0] == '3': # If the lines to specify facets have started
            line = [int(line[1]), int(line[2]), int(line[3])] # Convert the indices into integers
            pc_f.append(line)
pc_f = np.array(pc_f)
file1.close()

pcd_original = o3d.io.read_point_cloud(filename)

# Scale the point cloud into its original size
pc = np.asarray(pcd_original.points) * nerf_scale
pc_v = pc_v * nerf_scale 

pc_colors = np.asarray(pcd_original.colors)


print('Visualizing...')



# Create the window to pick up the desired point
vis_pick = o3d.visualization.VisualizerWithEditing()
vis_pick.create_window()

# Visualize the whole object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc)
pcd.colors = o3d.utility.Vector3dVector(pc_colors.astype(np.float64) / 255)

voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.01)
cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=3, radius=0.5)
voxel_down_pcd = voxel_down_pcd.select_by_index(ind)
#voxel_down_pcd.estimate_normals()

# Save the downsample point cloud (currently, it's still too dense)
o3d.io.write_point_cloud("downsample_" + filename, voxel_down_pcd)
                         
pc = np.array(voxel_down_pcd.points)
pc_colors = np.array(voxel_down_pcd.colors)
pc_tree = KDTree(pc)
# normals = np.array(voxel_down_pcd.normals)
print(len(pc))

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

# Display the object, so that the user can manually select one point

vis_pick.add_geometry(voxel_down_pcd)
vis_pick.add_geometry(frame)
vis_pick.run()
vis_pick.destroy_window()
p_sel_idx = np.asarray(vis_pick.get_picked_points()).astype('int')
print(p_sel_idx)
p_sel_candidates = np.asarray(voxel_down_pcd.points)[p_sel_idx]

##
## Generate grasp poses
##
# Consider only the upper half of the chair
# pc_upper_m = pc[:, 1] > -0.1
# pc_upper = pc[pc_upper_m]
# normals_upper = normals[pc_upper_m]

# Threshold of neighboring regions
th = 0.1


# Create the window to display grasp pose detection results
vis = o3d.visualization.Visualizer()
vis.create_window()


#p_sel_indices = [19196, 19672]
for p_sel in p_sel_candidates:
    # Build the gripper
    gripper = gripper_V_shape(0.5, 0.3, 0.1, 0.1, 0.1, 0.5, scale=0.2)
    gripper.open_gripper(np.pi/2)

    # Find a small region around the sampled point
    p_sel_neighbor_idx = pc_tree.query_radius(p_sel.reshape(1, -1), r=th)[0]
    if p_sel_neighbor_idx.shape[0] < 16: # NOt enough points in the local region
        continue
    # An approximate region selected by the human
    if p_sel[1] < -0.1:
        continue

    # A larger region for grasp pose detection
    p_sel_grasp_idx = pc_tree.query_radius(p_sel.reshape(1, -1), r=5 * th)[0]
    pc_neighbor = pc[p_sel_grasp_idx]



    # Fit the neighboring region to a quadratic surface -> in order to find 
    # the directions with principal curvatures
    c = fit_neighbor(pc, pcd, p_sel_neighbor_idx,visualization = False)
    df_axis_1, df_axis_2, p_sel_N_curvature, k1 , k2 , quadratic = \
        calculate_darboux_frame(c, p_sel, vis, visualization = False)
    
    
    # Move the gripper to that location
    df_axis_1, df_axis_2, p_sel_N_curvature, k1 , k2 , quadratic = \
            calculate_darboux_frame(c, p_sel, vis, visualization = True)
    # Mark the local neighboring region red
    for idx in p_sel_neighbor_idx:
        voxel_down_pcd.colors[idx] = [1.0, 0, 0]
    # Apply a transformation to fit the gripper to the darboux frame
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
    # 1. Rotate around y-axis (for pi/2)
    # 2. Elevate the gripper for 0.5
    # 3. Rotate around z-axis (for pi/2)
    df_tran1 = R.from_quat([0, np.sin(np.pi/4), 0, np.cos(np.pi/4)]).as_matrix()
    df_tran1 = np.vstack((np.hstack((df_tran1, np.array([[0],[0],[0]]))), np.array([0, 0, 0, 1])))
    df_tran2 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0.3],
        [0, 0, 0, 1]
    ])
    df_tran3 = R.from_quat([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)]).as_matrix()
    df_tran3 = np.vstack((np.hstack((df_tran3, np.array([[0],[0],[0]]))), np.array([0, 0, 0, 1])))
    df_tran_local = np.matmul(df_tran2, df_tran1)
    df_tran_local = np.matmul(df_tran3, df_tran_local)
    df_tran = np.matmul(darboux_frame, df_tran_local)
    gripper.apply_transformation(df_tran)

    generate_contact_graspnet_file(df_tran, pc_neighbor)

        
    gripper_pcd = []
    for part in gripper.parts:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(part)
        pcd.colors = o3d.utility.Vector3dVector(0.1 * np.ones(part.shape))
        vis.add_geometry(pcd)

vis.add_geometry(voxel_down_pcd)
vis.add_geometry(frame)
vis.run()
vis.destroy_window()



# grasp_candidates = []

# # Find a large region around the sample point for collision check
# check_collision_pc_idx = pc_tree.query_radius(p_sel.reshape(1, -1), r=0.2)

# # Do a grid search to generate several grasp candidates
# for x in range(4):
#     # Move the gripper a step forward (stepsize: 0.1)
#     tran_d = np.array([
#             [1, 0, 0, 0],
#             [0, 1, 0, 0],
#             [0, 0, 1, -0.05],
#             [0, 0, 0, 1]
#         ])
#     df_cand_translate = np.matmul(darboux_frame, np.matmul(tran_d, np.linalg.inv(darboux_frame)))
#     gripper.apply_transformation(df_cand_translate)

#     go_to_next_dist = False
#     for rot in range(6):
#         if go_to_next_dist == True:
#             # If we have found a good grasp pose at this distance, go to next distance
#             continue
#         # Rotate the gripper for an angle
#         rot_angle = np.pi/12
#         tran_rot = R.from_quat([0, 0, np.sin(rot_angle), np.cos(rot_angle)]).as_matrix()
#         tran_rot = np.vstack((np.hstack((tran_rot, np.array([[0],[0],[0]]))), np.array([0, 0, 0, 1])))
#         df_cand_rotate = np.matmul(darboux_frame, np.matmul(tran_rot, np.linalg.inv(darboux_frame)))
#         gripper.apply_transformation(df_cand_rotate)


#         collision = False
#         print("Calculate Min dist...")
#         for part in gripper.parts:
#             # Test the collision between each part of the gripper and the collision region

#             dists, fid, bc = pcu.closest_points_on_mesh(part.astype("float32", order='C'), pc_v, pc_f)
#             collision_dist = np.min(dists)
            
#             if collision_dist < 0.01: #Too close
#                 collision = True
#                 print("Collision Detected !")
#                 break 
#             else:
#                 continue
#         if collision: # If there is a collision, go to next candidate
#             continue
#         else: # If not, check whether the sampled point is within the gripper's grasping region
#             print("Point in Gripper?")
#             # If the point is outside the grasping region,
#             # it won't fall into the region again, even if you rotate it

#             # If the point is inside the grasping region,
#             # we have already obtained a good grasp pose, and no need to rotate it either
#             go_to_next_dist = True
            
#             in_gripper, quality = point_in_gripper(p_sel.flatten(), gripper=gripper)
#             if in_gripper:
#                 # If the point is within the grasping region, push it to the grasp candidates
#                 grasp_candidate_origin = gripper.frame[0:3, 3] # Origin of the gripper
#                 grasp_candidate_point_upper = gripper.upper_gripper[25*51-1] # Mid-point of the upper gripper
#                 grasp_candidate_point_lower = gripper.lower_gripper[25*51-1] # Mid-point of the lower gripper
#                 grasp_candidate = np.vstack((
#                     grasp_candidate_origin,
#                     grasp_candidate_point_upper,
#                     grasp_candidate_point_lower,
#                     np.array([quality, quality, quality])
#                 ))
#                 print("New grasp candidate")
#                 grasp_candidates.append(grasp_candidate)
#             else:
#                 continue


# vis.create_window()
# vis.add_geometry(pcd)
# vis.add_geometry(frame)

# # Visualization purpose
# df_axis_1, df_axis_2, p_sel_N_curvature, _, _, _ = calculate_darboux_frame(c, p_sel, vis, verbose = False)

# # Visualize the grasp candidates
# for candidate in grasp_candidates:
#     p1 = candidate[0, :]
#     p2 = candidate[1, :]
#     p3 = candidate[2, :]
#     color = candidate[3, 0]
#     grasp_candidate_points = [
#             [p1[0], p1[1], p1[2]],
#             [p2[0], p2[1], p2[2]],
#             [p3[0], p3[1], p3[2]]
#         ]
#     grasp_candidate_lines = [
#         [0, 1],
#         [0, 2]
#     ]
#     grasp_candidate_colors = [
#         [0.5, 0.5, 0],
#         [0.5, 0.5, 0]
#     ]


#     grasp_candidate_V = o3d.geometry.LineSet()
#     grasp_candidate_V.points = o3d.utility.Vector3dVector(grasp_candidate_points)
#     grasp_candidate_V.lines = o3d.utility.Vector2iVector(grasp_candidate_lines)
#     grasp_candidate_V.colors = o3d.utility.Vector3dVector(grasp_candidate_colors)
#     vis.add_geometry(grasp_candidate_V)


    