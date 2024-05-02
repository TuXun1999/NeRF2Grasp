import numpy as np
import open3d as o3d
from gripper import gripper_V_shape
from darboux_frame import fit_neighbor, calculate_darboux_frame
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import KDTree
from point_cloud_process import point_in_gripper, points_proj_to_plane, fit_cylinder_shell
import point_cloud_utils as pcu
import copy
import trimesh
## The main file to sample several grasp poses on the point cloud
## as well as evaulate the grasp qualities

nerf_scale = 0.64

'''
Section I: Read the whole object
'''
# Read the ply file
filename="chair.ply"
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

print(len(pc))
print('Visualizing...')


# Visualize the data
vis = o3d.visualization.Visualizer()
vis.create_window()

# Visualize the whole object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc)
pcd.colors = o3d.utility.Vector3dVector(pc_colors.astype(np.float64) / 255)

voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.005)
cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=16, radius=0.05)
voxel_down_pcd = voxel_down_pcd.select_by_index(ind)
# voxel_down_pcd.estimate_normals()

pc = np.array(voxel_down_pcd.points)
pc_colors = np.array(voxel_down_pcd.colors)
pc_tree = KDTree(pc)
# normals = np.array(voxel_down_pcd.normals)

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


'''
Sample several points on the object uniformly
Evaluate the geometric properties separately and 
Determine those that can be potential cylinder-shaped handles

==> TODO: use the sampling strategy to improve efficiency
'''
# Consider only the upper half of the chair
pc_upper_m = pc[:, 1] > -0.1
pc_upper = pc[pc_upper_m]
# normals_upper = normals[pc_upper_m]

# Threshold of neighboring regions
th = 0.1

# Uniformly sample 100 points on the upper half of the chair
pc_upper_num = len(pc_upper)
sample_point_num = 100
p_sel_upper_indices = np.random.randint(0, pc_upper_num, sample_point_num)



#p_sel_indices = [4666]
for p_sel_upper_idx in p_sel_upper_indices:
    #p_sel_idx = 4666 #np.random.randint(pc_num) #1859 #4666
    p_sel = pc_upper[p_sel_upper_idx]
    # Find a small region around the sampled point
    p_sel_neighbor_idx = pc_tree.query_radius(p_sel.reshape(1, -1), r=th)[0]
    if p_sel_neighbor_idx.shape[0] < 16: # NOt enough points in the local region
        continue
    # An approximate region selected by the human
    if p_sel[1] < -0.1:
        continue
    
    pc_neighbor = pc[p_sel_neighbor_idx]



    # Fit the neighboring region to a quadratic surface -> in order to find 
    # the directions with principal curvatures
    c = fit_neighbor(pc, pcd, p_sel_neighbor_idx,visualization = False)
    df_axis_1, df_axis_2, p_sel_N_curvature, k1 , k2 , quadratic = \
        calculate_darboux_frame(c, p_sel, vis, visualization = False)
    
    if (np.max((abs(k1), abs(k2))) < 50): 
        # Only consider the parts with enough curvatures & Close to a cylinder
        continue
    # Find the fitted cylinder
    pc_proj = points_proj_to_plane(p_sel, pc_neighbor, df_axis_1, df_axis_2, p_sel_N_curvature)
    hx, hy, r, error = fit_cylinder_shell(pc_proj)

    if error < 0.01: # If the fitting error is small
        df_axis_1, df_axis_2, p_sel_N_curvature, k1 , k2 , flat = \
             calculate_darboux_frame(c, p_sel, vis, visualization = True)
        # Mark the local neighboring region red
        for idx in p_sel_neighbor_idx:
            voxel_down_pcd.colors[idx] = [1.0, 0, 0]
        # Construct the transformation of Darboux Frame
        transformation = np.hstack((df_axis_1.reshape(-1,1), df_axis_2.reshape(-1,1),\
                                    p_sel_N_curvature.reshape(-1,1), p_sel.reshape(-1,1)))
        transformation = np.vstack((transformation, np.array([0, 0, 0, 1])))

        # Map the center of the cylinder back into the world frame
        cylinder_center = np.matmul(transformation, np.array([[0], [hx], [hy], [1]])).flatten()

        # Visualize the fitted cylinder
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=r,
                                                                height=th * 2)

        # Transform the cylinder to the desired place

        # At first, align the principal axis of the cylinder with the x-axis
        R = cylinder.get_rotation_matrix_from_xyz((0, np.pi/2, 0))
        cylinder.rotate(R, center = (0, 0, 0))

        # Apply the complete transformation
        cylinder_transform = copy.deepcopy(transformation)
        cylinder_transform[:, 3] = cylinder_center
        cylinder.transform(cylinder_transform)


        
        vis.add_geometry(cylinder)

vis.add_geometry(voxel_down_pcd)
vis.add_geometry(frame)
vis.run()
vis.destroy_window()
