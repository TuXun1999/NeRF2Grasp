import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
class gripper_V_shape():
    '''
    The class to initialize a V-shape gripper
    containing its point cloud
    '''
    def __init__(self, gl, gw, gh, r, ar, al, scale = 1.0):
        '''
        Initialize all the points
        '''
        ## The member to contain all geometric attributes of the gripper
        self.attributes = {}
        self.attributes["gripper_length"] = gl
        self.attributes["gripper_width"] = gw
        self.attributes["gripper_height"] = gh

        self.attributes["axis_radius"] = r
        self.attributes["arm_radius"] = ar
        self.attributes["arm_length"] = al

        for key, value in self.attributes.items():
            self.attributes[key] = value * scale
        
        self.attributes["scale"] = scale

        ## Construct the points (manually)
        self.rotation_axis = []
        
        # Construct the cylinder as the rotation axis
        for i in np.linspace(-gw/2, gw/2, 100):
            for ang in np.linspace(0, 2*np.pi, 60):
                self.rotation_axis.append([r * np.cos(ang), i, r * np.sin(ang)])
        
        self.upper_gripper = []
        # Construct half the gripper
        for i in np.linspace(-gw/2, gw/2, 51):
            for j in np.linspace(r, r + gl, 51):
                self.upper_gripper.append([j, i, 0])
        for k in np.linspace(gh/50, gh - gh/50, 49):
            for j in np.linspace(r, r + gl, 51):
                self.upper_gripper.append([j, gw/2, k])
            for i in np.linspace(gw/2 - gw/50, -gw/2 + gw/50, 49):
                self.upper_gripper.append([r + gl, i, k])
            for j in np.linspace(r, r + gl, 51):
                self.upper_gripper.append([j, -gw/2, k])
        for i in np.linspace(-gw/2, gw/2, 51):
            for j in np.linspace(r, r + gl, 51):
                self.upper_gripper.append([j, i, gh])
        
        
        self.lower_gripper = []
        # Construct half the gripper
        for i in np.linspace(-gw/2, gw/2, 51):
            for j in np.linspace(r, r + gl, 51):
                self.lower_gripper.append([j, i, 0])
        for k in np.linspace(-gh/50, -gh + gh/50, 49):
            for j in np.linspace(r, r + gl, 51):
                self.lower_gripper.append([j, gw/2, k])
            for i in np.linspace(gw/2 - gw/50, -gw/2 + gw/50, 49):
                self.lower_gripper.append([r + gl, i, k])
            for j in np.linspace(r, r + gl, 51):
                self.lower_gripper.append([j, -gw/2, k])
        for i in np.linspace(-gw/2, gw/2, 51):
            for j in np.linspace(r, r + gl, 51):
                self.lower_gripper.append([j, i, -gh])
        self.arm = []
        # Construct the arm of the gripper
        for i in np.linspace(-r, -r - al, 100):
            for ang in np.linspace(0, 2*np.pi, 40):
                self.arm.append([i, ar * np.cos(ang), ar * np.sin(ang)])

        # Convert them into numpy array (with scaling)
        self.rotation_axis = np.array(self.rotation_axis) * self.attributes["scale"]
        self.upper_gripper = np.array(self.upper_gripper) * self.attributes["scale"]
        self.lower_gripper = np.array(self.lower_gripper) * self.attributes["scale"]
        self.arm = np.array(self.arm) * self.attributes["scale"]

        self.parts = [self.rotation_axis, self.upper_gripper, self.lower_gripper, self.arm]

        # The local of the end-effector
        self.frame = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        # The opening angle of the gripper
        self.open_angle = 0
    def open_gripper(self, theta):
        '''
        Move the points so that the gripper is open to the angle "theta"
        '''
        rot_matrix1 =  R.from_quat([0, np.sin(-theta/4), 0, np.cos(-theta/4)]).as_matrix()
        rot_matrix2 =  R.from_quat([0, np.sin(theta/4), 0, np.cos(theta/4)]).as_matrix()

        self.upper_gripper = np.transpose(np.matmul(rot_matrix1, np.transpose(self.upper_gripper)))
        self.lower_gripper = np.transpose(np.matmul(rot_matrix2, np.transpose(self.lower_gripper)))
        
        self.parts[1] = self.upper_gripper
        self.parts[2] = self.lower_gripper

        self.open_angle = theta

    def init_from_file(self, file_name):
        '''
        The function to initialize the point cloud data from a file
        '''
        pass

    def apply_transformation(self, tran):
        '''
        The function to apply a transformation on the point cloud data
        Input:
        tran - the ordinary 4x4 transformation matrix 
        '''
        # for part in self.parts:
        #     part_points_num = part.shape[0]
        #     part = np.vstack((np.transpose(part), np.ones((1, part_points_num))))
        #     part = np.transpose(np.matmul(tran, part)[0:3])
        for p in range(len(self.parts)):
            part = self.parts[p]
            part_points_num = part.shape[0]
            part = np.vstack((np.transpose(part), np.ones((1, part_points_num))))
            self.parts[p] = np.transpose(np.matmul(tran, part)[0: 3])
        
        self.rotation_axis = self.parts[0]
        self.upper_gripper = self.parts[1]
        self.lower_gripper = self.parts[2]
        self.arm = self.parts[3]

        self.frame = np.matmul(tran, self.frame)