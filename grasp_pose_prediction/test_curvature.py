import trimesh
import open3d as o3d
from trimesh.curvature import discrete_gaussian_curvature_measure, \
    discrete_mean_curvature_measure
import numpy as np
# Specify the mesh file
filename= "./data/nerf/chair5_pm/chair_upper.obj"

# Read the file as a triangular mesh
mesh = o3d.io.read_triangle_mesh(filename)
mesh.compute_vertex_normals()
device = o3d.core.Device("CPU:0")
dtype = o3d.core.float32

# Create an empty point cloud
# Use pcd.point to access the points' attributes
pcd = o3d.t.geometry.PointCloud(device)

# Default attribute: "positions".
# This attribute is created by default and is required by all point clouds.
# The shape must be (N, 3). The device of "positions" determines the device
# of the point cloud.
pcd.point.positions = o3d.core.Tensor(np.array(mesh.vertices), dtype, device)

# Common attributes: "normals", "colors".
# Common attributes are used in built-in point cloud operations. The
# spellings must be correct. For example, if "normal" is used instead of
# "normals", some internal operations that expects "normals" will not work.
# "normals" and "colors" must have shape (N, 3) and must be on the same
# device as the point cloud.
pcd.point.normals = o3d.core.Tensor(np.array(mesh.vertex_normals), dtype, device)
pcd.point.colors = o3d.core.Tensor(np.array(mesh.vertex_colors), dtype, device)
boundarys, mask = pcd.compute_boundary_points(0.02, 30)
# TODO: not good to get size of points.
print(f"Detect {boundarys.point.positions.shape[0]} bnoundary points from {pcd.point.positions.shape[0]} points.")

boundarys = boundarys.paint_uniform_color([1.0, 0.0, 0.0])
pcd = pcd.paint_uniform_color([0.6, 0.6, 0.6])
o3d.visualization.draw_geometries([pcd.to_legacy(), boundarys.to_legacy()])

