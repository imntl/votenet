# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Provides Python helper function to read My SUNRGBD dataset.

Author: Charles R. Qi
Date: October, 2017

Updated by Charles R. Qi
Date: December, 2018
Note: removed basis loading.
'''
import numpy as np
import cv2
import os
import scipy.io as sio # to load .mat files for depth points

# Added for Blender
from skimage import io, transform

type2class={'ritzel':0, 'obj_1':1, 'obj_2':2, 'obj_3':3, 'obj_4':4, 'obj_5':5, 'obj_6':6, 'obj_7':7, 'obj_8':8, 'obj_9':9}
class2type = {type2class[t]:t for t in type2class}


def flip_axis_to_camera(pc):
    ''' Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
        Input and output are both (N,3) array
    '''
    pc2 = np.copy(pc)
    pc2[:,[0,1,2]] = pc2[:,[0,2,1]] # cam X,Y,Z = depth X,-Z,Y
    pc2[:,1] *= -1
    return pc2

def flip_axis_to_depth(pc):
    pc2 = np.copy(pc)
    pc2[:,[0,1,2]] = pc2[:,[0,2,1]] # depth X,Y,Z = cam X,Z,-Y
    pc2[:,2] *= -1
    return pc2


class BlenderObject3d(object):
    ''' labelXXXX.txt file has a new line for each object in scene.
        Each object has this data, seperatey by a space:
            classname - type of object
            cx - x coord of centroid
            cy - y coord of centroid
            cz - z coord of centroid
            w - width (half of the width?)
            l - length (half of the width?) - arrows in heading direction
            h - height (half of actual height?)
            ax - euler angel x
            ay - euler angel y
            az - euler angel z
    '''
    def __init__(self, line):
        data = line.split(' ')
        data[1:] = [float(x) for x in data[1:]]
        self.classname = data[0]
        self.cx = data[1]
        self.cy = data[2]
        self.cz = data[3]
        self.centroid = [self.cx,self.cy,self.cz]
        self.w = data[4]/3.75
        self.l = data[5]/3.75
        self.h = data[6]/3.9
        # Heading angle is radian with values between 0 and 2π for each euler angle
        self.heading_angle = [data[7],data[8],data[9]]
        self.heading_angle = [(i%(2*np.pi)) for i in self.heading_angle] 

class BLENDER_Calibration(object):
    ''' Calibration matrices and utils
        We define five coordinate system in Blender dataset

        camera coodinate:
            Z is forward, Y is downward, X is rightward

        depth coordinate:
            Just change axis order and flip up-down axis from camera coord

        upright depth coordinate: tilted depth coordinate by Rtilt such that Z is gravity direction,
            Z is up-axis, Y is forward, X is right-ward

        upright camera coordinate:
            Just change axis order and flip up-down axis from upright depth coordinate

        image coordinate:
            ----> x-axis (u)
           |
           v
            y-axis (v) 

        depth points are stored in upright depth coordinate.
        labels for 3d box (basis, centroid, size) are in upright depth coordinate.
        2d boxes are in image coordinate

        We generate frustum point cloud and 3d box in upright camera coordinate

        calibXXXX.txt - First line, 9 entries, equals scene matrix?!? (Rtilt!) 
                        Second line, 9 entries, equals camera matrix
                        Reshaped to 3x3 array of form:  1 4 7
                                                        2 5 8
                                                        3 6 9
    '''

    def __init__(self, calib_filepath):
        lines = [line.rstrip() for line in open(calib_filepath)] # Array with entries of each line as string
        Rtilt = np.array([float(x) for x in lines[0].split(' ')]) # First line as array with delimiter ' '
        self.Rtilt = np.reshape(Rtilt, (4,4), order='F') # Reshape first line to 3x3 array
        K = np.array([float(x) for x in lines[1].split(' ')]) # Second line as array wit delimiter ' '
        self.K = np.reshape(K, (3,3), order='F') # Reshape second line to 3x3 array
        self.f_u = self.K[0,0]
        self.f_v = self.K[1,1]
        self.c_u = self.K[0,2]
        self.c_v = self.K[1,2]

    def project_camera_to_global(self, pc):
        pc2 = np.dot(np.transpose(np.linalg.inv(self.Rtilt[0:3,0:3])),np.transpose(pc[:,0:3])) #(3,n)
        pc2 = np.transpose(pc2)
        pc2 = np.subtract(pc2,self.Rtilt[3,0:3])
        return pc2
   
    def project_upright_depth_to_camera(self, pc):
        ''' project point cloud from depth coord to camera coordinate
            Input: (N,3) Output: (N,3)
        '''
        # Project upright depth to depth coordinate
        pc2 = np.dot(np.transpose(self.Rtilt[0:3,0:3]), np.transpose(pc[:,0:3])) # (3,n)
        return flip_axis_to_camera(np.transpose(pc2))

    def project_upright_depth_to_image(self, pc):
        ''' Input: (N,3) Output: (N,2) UV and (N,) depth '''
        pc2 = self.project_upright_depth_to_camera(pc)
        uv = np.dot(pc2, np.transpose(self.K)) # (n,3)
        uv[:,0] /= uv[:,2]
        uv[:,1] /= uv[:,2]
        return uv[:,0:2], pc2[:,2]

    def project_upright_depth_to_upright_camera(self, pc):
        return flip_axis_to_camera(pc)

    def project_upright_camera_to_upright_depth(self, pc):
        return flip_axis_to_depth(pc)

    def project_image_to_camera(self, uv_depth):
        n = uv_depth.shape[0]
        x = ((uv_depth[:,0]-self.c_u)*uv_depth[:,2])/self.f_u
        y = ((uv_depth[:,1]-self.c_v)*uv_depth[:,2])/self.f_v
        pts_3d_camera = np.zeros((n,3))
        pts_3d_camera[:,0] = x
        pts_3d_camera[:,1] = y
        pts_3d_camera[:,2] = uv_depth[:,2]
        return pts_3d_camera

    def project_image_to_upright_camerea(self, uv_depth):
        pts_3d_camera = self.project_image_to_camera(uv_depth)
        pts_3d_depth = flip_axis_to_depth(pts_3d_camera)
        pts_3d_upright_depth = np.transpose(np.dot(self.Rtilt, np.transpose(pts_3d_depth)))
        return self.project_upright_depth_to_upright_camera(pts_3d_upright_depth)

 
 
def rotx(t):
    """Rotation about the x-axis."""
    if type(t) is not float:
        t = -t[1]
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def roty(t):
    """Rotation about the y-axis."""
    if type(t) is not float:
        t = t[0]
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rotz(t):
    """Rotation about the z-axis."""
    if type(t) is not float:
        t = t[2]
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def rotxyz(t):
    """Rotation about all 3 axis."""
    rot = np.dot(rotz(t), np.dot(roty(t), rotx(t)))
    return rot


def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def inverse_rigid_trans(Tr):
    """Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    """ 
    inv_Tr = np.zeros_like(Tr) # 3x4
    inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
    inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
    return inv_Tr

def read_blender_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)] # Array with each line as string entry
    objects = [BlenderObject3d(line) for line in lines] # Array with all objects of scene
    return objects

def load_depth_image(img_filename,calib):
    im = io.imread(img_filename)
    pic_3d = []
    for index_x, row in enumerate(im):
        for index_y, value in enumerate(row):
            z = (im[index_x,index_y]/65025)
            adj = (((calib.f_u + calib.f_v)/2)/((calib.c_u + calib.c_v)/2))
#            if z > 0.99: continue
            if z != 0:
                x = ((index_x - calib.c_u) * z / calib.f_u) * 1.25
                y = ((index_y - calib.c_v) * z / calib.f_v) * 1.25
                z = z * adj * 1.24
            else:
                x = 0
                y = 0
            pic_3d.append([x,y,-z])

    depth_pc = np.array(pic_3d)
    return calib.project_camera_to_global(depth_pc)

def load_depth_points(depth_filename):
    depth = np.loadtxt(depth_filename)
    return depth

def load_depth_points_mat(depth_filename):
    depth = sio.loadmat(depth_filename)['instance']
    return depth

def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds


def my_compute_box_3d(center, size, heading_angle):
    R = rotxyz(heading_angle)
    assert isRotationMatrix(R), "Rotationsmatrix kaputt"
    l,w,h = size
    x_corners = [-l,l,l,-l,-l,l,l,-l]
    y_corners = [w,w,-w,-w,w,w,-w,-w]
    z_corners = [h,h,h,h,-h,-h,-h,-h]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0,:] += center[0]
    corners_3d[1,:] += center[1]
    corners_3d[2,:] += center[2]
    return np.transpose(corners_3d)

def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def compute_box_3d(obj, calib):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in image coord.
            corners_3d: (8,3) array in in upright depth coord.
    '''
    center = obj.centroid

    # compute rotational matrix around yaw axis
    R = rotxyz(np.dot(-1,obj.heading_angle))
    #b,a,c = dimension
    #print R, a,b,c
    
    # 3d bounding box dimensions
    l = obj.l # along heading arrow
    w = obj.w # perpendicular to heading arrow
    h = obj.h

    # rotate and translate 3d bounding box
    x_corners = [-l,l,l,-l,-l,l,l,-l]
    y_corners = [w,w,-w,-w,w,w,-w,-w]
    z_corners = [h,h,h,h,-h,-h,-h,-h]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0,:] += center[0]
    corners_3d[1,:] += center[1]
    corners_3d[2,:] += center[2]

    # project the 3d bounding box into the image plane
    corners_2d,_ = calib.project_upright_depth_to_image(np.transpose(corners_3d))
    #print 'corners_2d: ', corners_2d
    return corners_2d, np.transpose(corners_3d)

def compute_orientation_3d(obj, calib):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        object orientation vector into the image plane.
        Returns:
            orientation_2d: (2,2) array in image coord.
            orientation_3d: (2,3) array in depth coord.
    '''
    
    # orientation in object coordinate system
    orientation_3d = np.array([[0,obj.heading_angle[0]],[0,obj.heading_angle[1]],[0,obj.heading_angle[2]]])
    center = obj.centroid
    orientation_3d[0,:] = orientation_3d[0,:] + center[0]
    orientation_3d[1,:] = orientation_3d[1,:] + center[1]
    orientation_3d[2,:] = orientation_3d[2,:] + center[2]
    
    # project orientation into the image plane
    orientation_2d,_ = calib.project_upright_depth_to_image(np.transpose(orientation_3d))
    return orientation_2d, np.transpose(orientation_3d)

def draw_projected_box3d(image, qs, color=(255,255,255), thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8,2) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    qs = qs.astype(np.int32)
    for k in range(0,4):
       #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
       i,j=k,(k+1)%4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA) # use LINE_AA for opencv3

       i,j=k+4,(k+1)%4 + 4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)

       i,j=k,k+4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
    return image


import pickle
import gzip

def save_zipped_pickle(obj, filename, protocol=-1):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object
