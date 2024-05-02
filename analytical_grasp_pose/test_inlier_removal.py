import open3d as o3d
import numpy as np
print("Convert mesh to a point cloud and estimate dimensions")
pcd = o3d.io.read_point_cloud("chair.ply")
diameter = 1.1 * np.linalg.norm(
    np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))

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
o3d.visualization.draw_geometries([pcd,frame])

print("Define parameters used for hidden_point_removal (at different locations)")
locations = [
    [0, 0, 0],
    [0, 0, diameter],
    [diameter, 0, 0],
    [0, diameter, 0],
    [0, 0, -diameter],
    [0, -diameter, 0],
    [-diameter, 0, 0]
]
pts = np.array([0, 0, 0])

for location in locations:
    camera = o3d.core.Tensor(location, o3d.core.float32)
    radius = diameter * 1000

    print("Get all points that are visible from given view point")
    pcd_from_legacy = o3d.t.geometry.PointCloud.from_legacy(pcd)
    _, pt_map = pcd_from_legacy.hidden_point_removal(camera, radius)

    pcd_from_legacy = pcd_from_legacy.select_by_index(pt_map)
    pcd_new = pcd_from_legacy.to_legacy()
    pts_part = np.array(pcd_new.points)
    pt_colors_part = np.array(pcd_new.colors)
    pts = np.vstack((pts, pts_part))


pcd = o3d.geometry.PointCloud()
print(pts)
pts = np.unique(pts, axis=0)
pcd.points = o3d.utility.Vector3dVector(pts)
pcd.colors = o3d.utility.Vector3dVector(np.ones(pts.shape) / 255)

print("Visualize result")
o3d.visualization.draw_geometries([pcd, frame])