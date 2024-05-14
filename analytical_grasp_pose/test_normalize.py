import open3d as o3d
import numpy as np
# Read the file as a triangular mesh
mesh = o3d.io.read_triangle_mesh("model_normalized.obj")
print(np.mean(np.asarray(mesh.vertices), axis=0))