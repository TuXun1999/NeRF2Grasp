import numpy as np
import open3d as o3d
import os
import json
import copy
import point_cloud_utils as pcu

'''
The file containing all the necessary mesh processing codes
'''
#############
# Part I: handle the mesh normalization
#############
def obj2stats(mesh):
    """
    Computes statistics of OBJ vertices and returns as {num,min,max,centroid}
    Input: mesh file read by obj
    """
    # Extract out the vertices of the msh
    vertices = np.asarray(mesh.vertices)

    # Find the maximum & minimum vertex, as well as other attributes
    minVertex = np.min(vertices, axis = 0)
    maxVertex = np.max(vertices, axis = 0)
    centroid = np.mean(vertices, axis = 0)
    numVertices = vertices.shape[0]

    info = {}
    info['numVertices'] = numVertices
    info['min'] = minVertex
    info['max'] = maxVertex
    info['centroid'] = centroid
    return info

def model_normalized(filename, out, normalize_stats_file = "normalize.npy", stats = None):
    '''
    The function to normalize the given mesh file
    '''
    """
    COPIED; Normalizes OBJ to be centered at origin and fit in unit cube
    """
    obj = o3d.io.read_triangle_mesh(filename)
    if not stats:
        stats = obj2stats(obj)

    # Extract out the necessary attributes
    diag = stats['max'] - stats['min']
    norm = np.linalg.norm(diag)
    c = stats['centroid']

    # Normalize the mesh
    obj_vertices = np.asarray(obj.vertices)
    obj_vertices = (obj_vertices - c) / norm

    # Store the normalized obj separately
    out_obj = copy.deepcopy(obj)
    out_obj.vertices = o3d.utility.Vector3dVector(obj_vertices)


    # Output the file
    o3d.io.write_triangle_mesh(out, out_obj)

    # Store the stats in a separate file
    with open(normalize_stats_file, 'wb') as f:
        np.save(f, stats, allow_pickle=True)
    return stats


def coordinate_correction(mesh, filename, nerf_scale):
    '''
    The function to convert the mesh into the correct coordinates
    Including: 
    1. Fix up the coordinate order 
    2. Scale it back to the correct range 
    3. Modify the original file
    '''
    mesh_coord_swap = np.asarray(mesh.vertices)
    mesh_coord_swap[:, [0, 1, 2]] = mesh_coord_swap[:, [2, 0, 1]]
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh_coord_swap) * nerf_scale)
    o3d.io.write_triangle_mesh(filename, mesh)
    return mesh

def read_normalize_stats(filename):
    '''
    Read the normalization stats stored in the json file directly
    '''
    return np.load(filename, allow_pickle=True)

###########
## Part II: Handle the grasp quality evaluation
###########
def collision_test(mesh, gripper_points, threshold):
    '''
    The function to check whether the gripper collides with the object mesh
    (globally evaluated; check the whole object mesh)
    '''
    # Method 1: use pcu to calculate the distances
    # pc_v = np.asarray(mesh.vertices)
    # pc_f = np.asarray(mesh.triangles)
    # dists, _, _ = pcu.closest_points_on_mesh(gripper_points.astype("float64", order='C'), pc_v, pc_f)
    # dist_min = np.min(dists)
    # # print(dist_min)
    # if dist_min < threshold:
    #     return True
    # else:
    #     return False
    
    # Method 2: use ray casting functionality from open3d 
    mesh_tri = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    # Create a scene and add the triangle mesh
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_tri)  # we do not need the geometry ID for mesh

    query_points = o3d.core.Tensor(gripper_points, dtype=o3d.core.Dtype.Float32)
    signed_distance = scene.compute_signed_distance(query_points).numpy()
    dist_min = np.min(signed_distance)
    if dist_min < threshold:
        return True
    else:
        return False
    

    
def collision_test_local(mesh, gripper_points, grasp_pose, gripper_attr, threshold, scale=1.5):
    '''
    The function to check whether the gripper collides with the object mesh
    (locally evaluated; create a bounding box around the gripper and only
    test collision inside the bounding box)
    '''
    ## Extract out the gripper parameters
    length = gripper_attr["Length"]
    width = gripper_attr["Width"]
    thickness = gripper_attr["Thickness"]

    # Specify the croping box vertices for a collision test
    pt1 = np.array([length, thickness, width]) * scale
    pt2 = np.array([-length, thickness, width]) * scale
    pt3 = np.array([-length, -thickness, width]) * scale 
    pt4 = np.array([length, -thickness, width]) * scale

    box_upper = np.vstack((pt1, pt2, pt3, pt4))
    box_lower = copy.deepcopy(box_upper)
    box_lower[:, 2] = -box_lower[:, 2]
    box = np.vstack((box_upper, box_lower))


    # Create the bounding box in open3d
    bbox = o3d.geometry.OrientedBoundingBox().create_from_points(o3d.utility.Vector3dVector(box))
    ## Transform the bbox to the grasp pose 
    # The rotation in open3d convention is along the center of the box,
    # not the origin of world frame
    # However, the center of the bounding box collides with the world frame!
    bbox_R = grasp_pose[0:3, 0:3]
    bbox_t = grasp_pose[0:3, 3]

    # Apply the transformations
    bbox.rotate(bbox_R)
    bbox.translate(bbox_t, relative=True)

    # Extract out the part of the object in the region for collision
    mesh_collision_region = mesh.crop(bbox)

    if mesh_collision_region.is_empty():
        return False, bbox, mesh_collision_region
    ## Find out the minimum dists
    # Method 1: Use pcu library
    # pc_v = np.asarray(mesh_collision_region.vertices)
    # pc_f = np.asarray(mesh_collision_region.triangles)
    # dists, _, _ = pcu.closest_points_on_mesh(gripper_points.astype("float64", order='C'), pc_v, pc_f)
    # dist_min = np.min(dists)
    # # print(dist_min)
    # if dist_min < threshold:
    #     bbox.color = (1.0, 0, 0)
    #     return True, bbox, mesh_collision_region
    # else:
    #     bbox.color = (0, 1, 0)
    #     return False, bbox, mesh_collision_region
    
    # Method 2: Use RayCast scene in open3d
    # Create a scene and add the triangle mesh
    mesh_tri = o3d.t.geometry.TriangleMesh.from_legacy(mesh_collision_region)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_tri)  # we do not need the geometry ID for mesh

    query_points = o3d.core.Tensor(gripper_points, dtype=o3d.core.Dtype.Float32)
    signed_distance = scene.compute_signed_distance(query_points).numpy()
    dist_min = np.min(signed_distance)
    if dist_min < threshold:
        return True, bbox, mesh_collision_region
    else:
        return False, bbox, mesh_collision_region

def antipodal_test(mesh, grasp_pose, gripper_attr, k, theta):
    '''
    The function to evaluate whether a grasp is valid based on antipodal metric
    '''
    ## Determine the closing region
    # Extract out the gripper parameters
    length = gripper_attr["Length"]
    width = gripper_attr["Width"]
    thickness = gripper_attr["Thickness"]

    # Specify the box vertices
    pt1 = np.array([0, thickness/2, width/2])
    pt2 = np.array([-length, thickness/2, width/2])
    pt3 = np.array([-length, -thickness/2, width/2])
    pt4 = np.array([0, -thickness/2, width/2])

    box_upper = np.vstack((pt1, pt2, pt3, pt4))
    box_lower = copy.deepcopy(box_upper)
    box_lower[:, 2] = -box_lower[:, 2]
    box = np.vstack((box_upper, box_lower))



    # Create the bounding box in open3d
    bbox = o3d.geometry.OrientedBoundingBox().create_from_points(o3d.utility.Vector3dVector(box))
    # Transform the bbox to the grasp pose 
    # bbox.transform(grasp_pose) (transform is not supported??)
    # The rotation in open3d convention is along the center of the box,
    # not the origin of world frame
    bbox_center_w = np.array([-length/2, 0, 0])
    bbox_R = grasp_pose[0:3, 0:3]
    bbox_t = grasp_pose[0:3, 3]

    # Calculate the relative distance
    rel = np.matmul((np.eye(3) - bbox_R), bbox_center_w)
    bbox.rotate(bbox_R)
    # Correction on the different rotation axes
    bbox.translate(-rel)
    bbox.translate(bbox_t, relative=True)
    bbox.color = (0.0, 0.5, 0.5)

    ## Extract out the part of the object in the closing region
    mesh_closing_region = mesh.crop(bbox)
    if mesh_closing_region.is_empty(): # Return false directly if no region is extracted out
        return False, bbox
    mesh_closing_region.compute_vertex_normals()
    mesh_normals = np.asarray(mesh_closing_region.triangle_normals)
    
    ## Antipodal Metric: at least k normals deviate from the closing vector by at most theta
    # Find the direction of gripper closing vectors
    close_vector_up = np.matmul(grasp_pose, np.array([
        [0], [0], [1], [0]
    ])).flatten()[:-1]


    # Compute the dot product 
    antipodal_up = mesh_normals.dot(close_vector_up)

    # Count how many normals are aligned with the closing vector
    parallel = (antipodal_up > np.cos(theta)).sum()
    anti_parallel = (antipodal_up < -np.cos(theta)).sum()

    return (parallel >= k and anti_parallel >= k), bbox

    



##########
## Part III: Handle the depth scale ambiguity (to find the point in space)
##########
def point_select_in_space(camera_pose, ray_dir, mesh):
    '''
    The function to determine the location of the selected point on the image in space
    (Raycasting functionality used)
    '''
    # Create a scene & Add the mesh to the scene
    scene = o3d.t.geometry.RaycastingScene()
    mesh_id = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))


    # Cast the ray from the camera
    camera_location = camera_pose[0:3, 3]
    ray = list(np.hstack((camera_location, ray_dir)))
    rays = o3d.core.Tensor([ray],
                       dtype=o3d.core.Dtype.Float32)

    ans = scene.cast_rays(rays)

    # Obtain the distance upon hitting
    point_select_distance = min(ans['t_hit'].numpy()[0], 200)

    # Calculate the 3D point coordinates
    pos = camera_pose[0:3, 3] + ray_dir * point_select_distance
    return pos, point_select_distance