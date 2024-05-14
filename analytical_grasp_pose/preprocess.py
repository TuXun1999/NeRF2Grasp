import numpy as np
import open3d as o3d
import os
import json
import copy
'''
The file containing all the necessary preprocessing codes
'''
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
    np.save(normalize_stats_file, stats, allow_pickle=True)
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
