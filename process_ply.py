import numpy as np
import open3d as o3d

 # read ply file
pcd = o3d.io.read_point_cloud('chair.ply')
a = np.asarray(pcd.points)
c = np.asarray(pcd.colors)
a_flipped = np.ones(a.shape)
a_flipped[:, 0] = a[:, 0]
a_flipped[:, 1] = a[:, 2]
a_flipped[:, 2] = a[:, 1]
print(a[0])
print(c[0] * 255)

npy_data = {}
npy_data['xyz_color'] = c
npy_data['xyz'] = a_flipped
camera_intrinsic = np.array([
    [400, 0, 400],
    [0, 400, 400],
    [0, 0, 1]
])
npy_data['K'] = camera_intrinsic

np.save("point_cloud_test1", npy_data)