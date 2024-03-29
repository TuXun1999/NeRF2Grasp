import point_cloud_utils as pcu
import numpy as np
import open3d as o3d
'''
Test file to see how to also extract the facets on the mesh
'''
filename = "fuze.ply"
pc_v, pc_f = pcu.load_mesh_vf(filename)

if pc_f is None:
    pc_f = []
    # Using readlines()
    file1 = open(filename, 'r')
    lines = file1.readlines()
    
    # Strips the newline character
    for line in lines:
        line = line.split()
        if line[0] == '3': # If the lines to specify facets have started
            line = [int(line[1]), int(line[2]), int(line[3])] # Convert the indices into integers
            pc_f.append(line)
pc_f = np.array(pc_f)

# Generate random points on a sphere around the shape
part = np.random.randn(3300, 3)
part /= np.linalg.norm(part, axis=-1, keepdims=True)
part = part.astype("float32")
# Visualize the data
vis = o3d.visualization.Visualizer()

# Visualize the used points
model_pcd= o3d.geometry.PointCloud()
model_pcd.points = o3d.utility.Vector3dVector(pc_v)
model_pcd.colors = o3d.utility.Vector3dVector(np.ones(pc_v.shape) / 255)

sample_pcd = o3d.geometry.PointCloud()
sample_pcd.points = o3d.utility.Vector3dVector(part)
sample_points_color =  np.zeros(part.shape)
sample_points_color[:, 0] = 1
sample_pcd.colors = o3d.utility.Vector3dVector(sample_points_color)


vis.create_window()
vis.add_geometry(model_pcd)
vis.add_geometry(sample_pcd)
print(pc_f)
print(part.dtype)
print(pc_v.dtype)
vis.run()
vis.destroy_window()
dists, fid, bc = pcu.closest_points_on_mesh(part, pc_v, pc_f)
collision_dist = np.min(dists)

print(collision_dist)
