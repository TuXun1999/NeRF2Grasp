from skimage.util import view_as_blocks
import numpy as np
image = "chair_sim_depth.density_slices_512x512x512.png"
res = image.shape[0:2]
grid_3d = view_as_blocks(image, (res[1], res[0]))/255.0
grid_3d = grid_3d.reshape(-1, res[1], res[0])[:res[2], :, :].T 

threshold = 0.1
vertices = np.array(np.where(grid_3d > threshold)).T.astype(float)
print(vertices.shape)
