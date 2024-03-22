import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

import matplotlib as mpl
import math
import scipy
import open3d as o3d

def fn(x, y, z, c):
    return c[0] * x * x + c[1] * y * y + c[2] * z * z + \
            c[3] * x * y + c[4] * x * z + c[5] * y * z + \
            c[6] * x + c[7] * y + c[8] * z + \
            c[9]

# Read the npy file
pc_file = np.load("point_cloud_fuze.npy", allow_pickle=True)

pc_file = pc_file.item()
pc = pc_file['xyz']
pc_colors = pc_file['xyz_color']
print(len(pc))

pc_num = len(pc)
p_sel_idx =  np.random.randint(pc_num) # Bad example: 1497, 2736
p_sel = pc[p_sel_idx]
print(p_sel_idx)

print('Visualizing...')
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc)
pcd.colors = o3d.utility.Vector3dVector(pc_colors.astype(np.float64) / 255)

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)






M_fit = np.zeros((10, 10))
N_fit = np.zeros((10, 10))

p_sel_neighbor_idx = []
p_sel_neighbor_th = 0.05

# Find a small region around the sampled point
for i in range(pc_num):
    p_neighbor_cand = pc[i]
    dist = math.sqrt((p_neighbor_cand[0] - p_sel[0]) ** 2 + \
                    (p_neighbor_cand[1] - p_sel[1]) ** 2 + \
                    (p_neighbor_cand[2] - p_sel[2]) ** 2)
    if (dist < p_sel_neighbor_th):
        p_sel_neighbor_idx.append(i)
p_sel_neighbor_idx = np.array(p_sel_neighbor_idx)

## Section I: find a quadratic surface to fit to the points
for i in p_sel_neighbor_idx:
    # Recolor the points in red
    pcd.colors[i] = [1.0, 0.0, 0.0]
    p = pc[i]
    px = p[0]
    py = p[1]
    pz = p[2]


    l = np.array([
        [px * px],
        [py * py],
        [pz * pz],
        [px * py],
        [px * pz],
        [py * pz],
        [px],
        [py],
        [pz],
        [1]
    ])

    lx = np.array([
        [2 * px],
        [0],
        [0],
        [py],
        [pz],
        [0],
        [1],
        [0],
        [0],
        [0]
    ])

    ly = np.array([
        [0],
        [2 * py],
        [0],
        [px],
        [0],
        [pz],
        [0],
        [1],
        [0],
        [0]
    ])

    lz = np.array([
        [0],
        [0],
        [2 * pz],
        [0],
        [px],
        [py],
        [0],
        [0],
        [1],
        [0]
    ])
    # required calculations for M
    M_fit = M_fit + np.matmul(l, np.transpose(l))


    # required calculations for N
    N_fit = N_fit + \
        np.matmul(lx, np.transpose(lx)) + \
            np.matmul(ly, np.transpose(ly)) + \
                np.matmul(lz, np.transpose(lz))

# Obtain the parameters to fit the point cloud to a quadratic surface
eig_values, eig_vectors = scipy.linalg.eig(M_fit, N_fit)

c = eig_vectors[:, np.argmin(eig_values)] 

print("===========")
print("Test Fitting Quadratic Surface")
print(np.matmul(np.transpose(c), np.matmul(M_fit, c)))
x,y,z = pc[p_sel_neighbor_idx[np.random.randint(len(p_sel_neighbor_idx))]]
print(fn(x,y,z,c))
print("==============")

## Section II: Obtain the normal vector at that point
#Idea 1: Just obtain the normal at that point
fx = 2* c[0] * p_sel[0] + c[3] * p_sel[1] + c[4] * p_sel[2] + c[6]
fy = 2* c[1] * p_sel[1] + c[3] * p_sel[0] + c[5] * p_sel[2] + c[7]
fz = 2* c[2] * p_sel[2] + c[4] * p_sel[0] + c[5] * p_sel[1] + c[8]
f = [fx, fy, fz]
p_sel_N = np.array([
        [fx],
        [fy],
        [fz]
    ])
p_sel_N = p_sel_N / np.linalg.norm(p_sel_N)

# Idea 2: Find the average normals
# p_sel_normals = np.zeros((3, len(p_sel_neighbor)))
# for i in range(len(p_sel_neighbor)):
#     p_neighbor = p_sel_neighbor[i]
#     p_neighbor_normal = np.array([
#         [2* c[0] * p_neighbor[0] + c[3] * p_neighbor[1] + c[4] * p_neighbor[2] + c[6]],
#         [2* c[1] * p_neighbor[1] + c[3] * p_neighbor[0] + c[5] * p_neighbor[2] + c[7]],
#         [2* c[2] * p_neighbor[2] + c[4] * p_neighbor[0] + c[5] * p_neighbor[1] + c[8]]
#     ])
#     p_neighbor_normal = p_neighbor_normal/np.linalg.norm(p_neighbor_normal)

#     p_sel_normals[:, i] = p_neighbor_normal.flatten()


# p_sel_N = np.mean(p_sel_normals, axis=1)

## Section III: Find the Darboux Frame
df = np.array([
    [2 * c[0], c[3] , c[4]],
    [c[3], 2 * c[1], c[5]],
    [c[4], c[5], 2 * c[2]]
])
# Idea 1: Analytical method
# Temporarily Abandoned

# Idea 2: Use (I - NN^T)\gradient N to approximate the Darboux frame --> may be inaccurate

dN = np.zeros((3,3))
for a in range(3):
    for b in range(3):
        lower = fx * fx + fy * fy + fz * fz
        upper = df[a][b] * math.sqrt(lower)- \
            (f[a]/math.sqrt(lower)) * (fx * df[0][b] + fy * df[1][b] + fz * df[2][b])
        dN[a][b] = upper/lower

eval_matrix = (np.eye(3) - np.matmul(p_sel_N, np.transpose(p_sel_N)))
eval_matrix = np.matmul(eval_matrix, dN)
# Find the curvature at the selected point
eig_values, eig_vectors = scipy.linalg.eig(eval_matrix)

df_axis_1 = eig_vectors[:, np.argmin(eig_values)]
df_axis_2 = eig_vectors[:, np.argmax(eig_values)]

print("===========")
print("Test Darboux Frame")
print(p_sel_N)
print(df_axis_1)
print(df_axis_2)
print(np.matmul(df_axis_1, p_sel_N))
print(np.matmul(df_axis_2, p_sel_N))
print(np.matmul(np.transpose(df_axis_1), df_axis_2))



# Draw out the normal vector & the Darboux Frame
p_sel_N_vis = p_sel + 0.2 * p_sel_N.flatten()


p_sel_d1_vis = p_sel + 0.2 * df_axis_1
p_sel_d2_vis = p_sel + 0.2 * df_axis_2
darboux_frame_points = [
    [p_sel[0], p_sel[1], p_sel[2]],
    [p_sel_N_vis[0], p_sel_N_vis[1], p_sel_N_vis[2]],
    [p_sel_d1_vis[0], p_sel_d1_vis[1], p_sel_d1_vis[2]],
    [p_sel_d2_vis[0], p_sel_d2_vis[1], p_sel_d2_vis[2]]
]
darboux_frame_lines = [
    [0, 1],
    [0, 2],
    [0, 3]
]
darboux_frame_colors = [
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0]
]


darboux_frame = o3d.geometry.LineSet()
darboux_frame.points = o3d.utility.Vector3dVector(darboux_frame_points)
darboux_frame.lines = o3d.utility.Vector2iVector(darboux_frame_lines)
darboux_frame.colors = o3d.utility.Vector3dVector(darboux_frame_colors)
vis.add_geometry(darboux_frame)
vis.run()
vis.destroy_window()

