import numpy as np

import math
import scipy
import open3d as o3d

def fn(x, y, z, c):
    return c[0] * x * x + c[1] * y * y + c[2] * z * z + \
            c[3] * x * y + c[4] * x * z + c[5] * y * z + \
            c[6] * x + c[7] * y + c[8] * z + \
            c[9]


def partial_derivatives(fx_o, fy_o, fz_o, df, orientation):
    '''
    The function to calculate the partial derivatives
    hx, hy, hxx, hxy, hyy
    '''
    if orientation == "z": # x, y --> z
        fx = fx_o
        fy = fy_o
        fz = fz_o
        fxx = df[0, 0]
        fxy = df[0, 1]
        fxz = df[0, 2]
        fyy = df[1, 1]
        fyz = df[1, 2]
        fzz = df[2, 2]
    elif orientation == "x": #y, z --> x
        fx = fy_o
        fy = fz_o 
        fz = fx_o
        fxx = df[1, 1]
        fxy = df[1, 2]
        fxz = df[1, 0]
        fyy = df[2, 2]
        fyz = df[2, 0]
        fzz = df[0, 0]
    elif orientation == "y": #x, z --> y
        fx = fx_o
        fy = fz_o
        fz = fy_o
        fxx = df[0, 0]
        fxy = df[0, 2]
        fxz = df[0, 1]
        fyy = df[2, 2]
        fyz = df[2, 1]
        fzz = df[1, 1]

    
    hx = - (fx/fz)
    hy = - (fy/fz)
    hxx = (2*fx*fz*fxz - fx*fx*fzz - fz*fz*fxx)/(fz*fz*fz)
    hxy = (fx*fz*fyz + fy*fz*fxz - fx*fy*fzz - fz*fz*fxy)/(fz*fz*fz)
    hyy = (2*fy*fz*fyz - fy*fy*fzz - fz*fz*fyy)/(fz*fz*fz)
    return hx, hy, hxx, hxy, hyy

def shape_operator_coefficients(hx, hy, hxx, hxy, hyy):
    '''
    The function to calculate the shape operator coefficients:
    L, M, N, E, F, G
    '''
    E = 1 + hx * hx
    F = hx * hy
    G = 1 + hy * hy
    L = hxx / math.sqrt(1 + hx * hx + hy * hy)
    M = hxy / math.sqrt(1 + hx * hx + hy * hy)
    N = hyy / math.sqrt(1 + hx * hx + hy * hy)
    return L, M, N, E, F, G

def shape_operator(L, M, N, E, F, G):
    '''
    The function to calculate the shape operator
    P = [L M; M N] * [E F; F G]^{-1}
    '''
    P1 = np.array([[L, M], [M, N]])
    P2 = np.array([[E, F], [F, G]])
    P = np.matmul(P1, np.linalg.inv(P2))

    print("=====================")
    print("Test the darboux frame calculation")
    print("==================")
    K = (L * N - M * M)/(E * G - F * F)
    H = (G * L - 2 * F * M + E * N)/(2 * (E * G - F * F))
    k1 = H + math.sqrt(H*H - K)
    k2 = H - math.sqrt(H*H - K)
    print("Curvature values")
    print(k1)
    print(k2)
    lambda1 = (k1 * F - M)/(N - k1 * G)
    lambda2 = (k2 * E-  L)/(M- k2 * F)
    print("Lambda values")
    print(lambda1)
    print(lambda2)
    bt = E * N - G * L 
    at = F * N - G * M 
    ct = E * M - F * L 
    delta_t = bt*bt - 4 *at * ct
    print("Test values for Lambda")
    print((-bt + math.sqrt(delta_t))/(2*at))
    print((-bt - math.sqrt(delta_t))/(2*at))
    return P


def find_darboux_frame(P, hx, hy, orientation):
    '''
    The function to find the darboux frame at the specified point
    '''
    eig_values, eig_vectors = scipy.linalg.eig(P)
    df_dxy_1 = eig_vectors[:, np.argmin(eig_values)]
    df_dxy_2 = eig_vectors[:, np.argmax(eig_values)]
    print(eig_values)
    print(eig_vectors)
    df_dz_1 = -hx * df_dxy_1[1] + hy * df_dxy_1[0]
    df_dz_2 = - hx * df_dxy_2[1] + hy * df_dxy_2[0]
    if orientation == "z":
        df_axis_1 = np.array([-df_dxy_1[1], df_dxy_1[0], df_dz_1])
        df_axis_1 = df_axis_1 / np.linalg.norm(df_axis_1)
        df_axis_2 = np.array([-df_dxy_2[1], df_dxy_2[0], df_dz_2])
        df_axis_2 = df_axis_2 / np.linalg.norm(df_axis_2)
    elif orientation == "x":
        df_axis_1 = np.array([-df_dz_1, df_dxy_1[1], df_dxy_1[0]])
        df_axis_1 = df_axis_1 / np.linalg.norm(df_axis_1)
        df_axis_2 = np.array([-df_dz_2, df_dxy_2[1], df_dxy_2[0]])
        df_axis_2 = df_axis_2 / np.linalg.norm(df_axis_2)
    elif orientation == "y":
        df_axis_1 = np.array([-df_dxy_1[1], df_dz_1, df_dxy_1[0]])
        df_axis_1 = df_axis_1 / np.linalg.norm(df_axis_1)
        df_axis_2 = np.array([-df_dxy_2[1], df_dz_2, df_dxy_2[0]])
        df_axis_2 = df_axis_2 / np.linalg.norm(df_axis_2)
    print(np.max(eig_values))
    print(np.min(eig_values))
    
    return df_axis_1, df_axis_2

# Read the npy file
pc_file = np.load("point_cloud_fuze.npy", allow_pickle=True)

pc_file = pc_file.item()
pc = pc_file['xyz']
pc_colors = pc_file['xyz_color']
print(len(pc))

pc_num = len(pc)
p_sel_idx = 2736 #np.random.randint(pc_num) #1080 #1570 #142
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
# Find \gradient N
# df[i][j] = \frac{d f_i}{d x_j}

print("Partial derivatives")
print(fx)
print(fy)
print(fz)
if abs(fz) >= 1e-4:
    hx, hy, hxx, hxy, hyy = partial_derivatives(fx, fy, fz, df, "z")
    L, M, N, E, F, G = shape_operator_coefficients(hx, hy, hxx, hxy, hyy)
    P = shape_operator(L, M, N, E, F, G)
    df_axis_1, df_axis_2 = find_darboux_frame(P, hx, hy, "z")
    p_sel_N_curvature = np.array([-hx, -hy, 1])
    
elif abs(fx) >= 1e-4:
    hx, hy, hxx, hxy, hyy = partial_derivatives(fx, fy, fz, df, "x")
    L, M, N, E, F, G = shape_operator_coefficients(hx, hy, hxx, hxy, hyy)
    P = shape_operator(L, M, N, E, F, G)
    df_axis_1, df_axis_2 = find_darboux_frame(P, hx, hy, "x")
    p_sel_N_curvature = np.array([1, -hx, -hy])
elif abs(fy) >= 1e-4:
    hx, hy, hxx, hxy, hyy = partial_derivatives(fx, fy, fz, df, "y")
    L, M, N, E, F, G = shape_operator_coefficients(hx, hy, hxx, hxy, hyy)
    P = shape_operator(L, M, N, E, F, G)
    df_axis_1, df_axis_2 = find_darboux_frame(P, hx, hy, "y")
    p_sel_N_curvature = np.array([-hx, 1, -hy])
else:
    print("Error! fx, fy, fz are all 0; singular point")
    # Use (I - N N^T) \grad N instead
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
    p_sel_N_curvature = p_sel_N.flatten()

p_sel_N_curvature = p_sel_N_curvature / np.linalg.norm(p_sel_N_curvature)
# Idea 2: Use (I - NN^T)\gradient N to approximate the Darboux frame --> may be inaccurate

print("===========")
print("Test Darboux Frame")
print(p_sel_N)
print(df_axis_1)
print(df_axis_2)
print(np.matmul(df_axis_1, p_sel_N))
print(np.matmul(df_axis_2, p_sel_N))
print(np.matmul(np.transpose(df_axis_1), df_axis_2))



# Draw out the normal vector & the Darboux Frame
p_sel_N_curvature_vis = p_sel + 0.2 * p_sel_N_curvature

p_sel_d1_vis = p_sel + 0.2 * df_axis_1
p_sel_d2_vis = p_sel + 0.2 * df_axis_2
darboux_frame_points = [
    [p_sel[0], p_sel[1], p_sel[2]],
    [p_sel_N_curvature_vis[0], p_sel_N_curvature_vis[1], p_sel_N_curvature_vis[2]],
    [p_sel_d1_vis[0], p_sel_d1_vis[1], p_sel_d1_vis[2]],
    [p_sel_d2_vis[0], p_sel_d2_vis[1], p_sel_d2_vis[2]]
]
darboux_frame_lines = [
    [0, 1],
    [0, 2],
    [0, 3],
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

