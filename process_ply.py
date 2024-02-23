import numpy as np
import open3d as o3d

 # read ply file
pcd = o3d.io.read_point_cloud('chair.ply')
a = np.asarray(pcd.points)
c = np.asarray(pcd.colors)
print(a[0])
print(c[0] * 255)

npy_data = {}
npy_data['xyz_color'] = c
npy_data['xzy'] = a
camera_intrinsic = np.array([
    [400, 0, 400],
    [0, 400, 400],
    [0, 0, 1]
])
npy_data['K'] = camera_intrinsic

np.save("point_cloud_test1", npy_data)