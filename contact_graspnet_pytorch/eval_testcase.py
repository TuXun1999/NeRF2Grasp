
import numpy as np


testcase_data = np.load("./test_data/0.npy", allow_pickle=True)
testcase_data = testcase_data.item()
rgb = testcase_data['rgb']
depth = testcase_data['depth']
intrinsic = testcase_data['K']
seg = testcase_data['seg']
print(rgb.shape)
print(seg.shape)
print(depth)
print(depth.shape)