import point_cloud_utils as pcu

'''
Test file to see how to also extract the facets on the mesh
'''
pc_v, pc_f = pcu.load_mesh_vf("fuze.ply")
print(pc_f)