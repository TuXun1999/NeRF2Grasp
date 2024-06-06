import open3d as o3d
import open3d.visualization as vis
import numpy as np

sphere = o3d.geometry.TriangleMesh.create_sphere(1.0)
sphere.compute_vertex_normals()
sphere.translate(np.array([0, 0, -3.5]))
box = o3d.geometry.TriangleMesh.create_box(2, 4, 4)
box.translate(np.array([-1, -2, -2]))
box.compute_triangle_normals()

mat_sphere = vis.rendering.MaterialRecord()
mat_sphere.shader = 'defaultLit'
mat_sphere.base_color = [0.8, 0, 0, 1.0]

mat_box = vis.rendering.MaterialRecord()
# mat_box.shader = 'defaultLitTransparency'
mat_box.shader = 'defaultLitSSR'
mat_box.base_color = [0.467, 0.467, 0.467, 0.2]
mat_box.base_roughness = 0.0
mat_box.base_reflectance = 0.0
mat_box.base_clearcoat = 1.0
mat_box.thickness = 1.0
mat_box.transmission = 0.2
mat_box.absorption_distance = 10
mat_box.absorption_color = [0.5, 0.5, 0.5]

geoms = [{'name': 'sphere', 'geometry': sphere, 'material': mat_sphere},\
        {'name': 'box', 'geometry': box, 'material': mat_box}]

vis.draw(geoms)

