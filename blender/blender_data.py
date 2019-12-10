# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Helper class and functions for loading SUN RGB-D objects

Author: Charles R. Qi
Date: December, 2018

Note: removed unused code for frustum preparation.
Changed a way for data visualization (removed depdency on mayavi).
Load depth with scipy.io
'''

import os
import sys
import numpy as np
import sys
import cv2
import argparse
from PIL import Image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils/'))
import pc_util
import blender_utils

from tqdm import tqdm 

import plyfile
import open3d as o3d

DEFAULT_TYPE_WHITELIST = ['ritzel','obj_1','obj_2','obj_3','obj_4','obj_5','obj_6','krones_1','krones_2','krones_3']

class blender_object(object):
    ''' Load and parse object data '''
    def __init__(self, root_dir, split='train'): # Root dir shold be a directory with train and test
        self.data_path = root_dir
        self.split = split
        assert self.split == 'train' or self.split == 'test', "Split should be train or test"
        self.split_dir = os.path.join(root_dir, split)

        self.index = []
        for foldername in os.listdir(self.split_dir):
            if not foldername.endswith(".txt"):
                for filename in os.listdir(os.path.join(self.split_dir, foldername)):
                    if filename.startswith("depth"):
                        self.index.append([int(foldername),int('{:10.4}'.format(filename[5:]))])

    def __len__(self):
        return len(self.index)

    def get_calibration(self, idx):
        calib_filename = os.path.join(self.split_dir, "{:02d}".format(self.index[idx][0]), "calib{:04d}.txt".format(self.index[idx][1]))
        return blender_utils.BLENDER_Calibration(calib_filename)

    def get_depth(self, idx): 
        depth_filename = os.path.join(self.split_dir, "{:02d}".format(self.index[idx][0]), "depth{:04d}.png".format(self.index[idx][1]))
        calib = self.get_calibration(idx)
        return blender_utils.load_depth_image(depth_filename,calib)

    def get_label_objects(self, idx):
        label_filename = os.path.join(self.split_dir, "{:02d}".format(self.index[idx][0]), "label{:04d}.txt".format(self.index[idx][1]))
        return blender_utils.read_blender_label(label_filename)

def extract_blender_data(data_dir, split = 'train', num_point=20000, # Extracs data to .npz and .npy
    type_whitelist=DEFAULT_TYPE_WHITELIST, save_votes=True):
    """ Extract scene point clouds and 
    bounding boxes (centroids, box sizes, heading angles, semantic classes).
    Dumped point clouds and boxes are in upright depth coord.

    Args:
        root_dir: folder of the dataset
        split: training or testing
        save_votes: whether to compute and save Ground truth votes.

    Dumps:
        <id>_pc.npz of (N,3) where N is for number of subsampled points and 3 is
            for XYZ in upright depth coord
        <id>_bbox.npy of (K,8) where K is the number of objects, 10 is for
            centroids (cx,cy,cz), dimension (l,w,h), heanding_angle (ax,ay,az) and semantic_class
        <id>_votes.npz of (N,10) with 0/1 indicating whether the point belongs to an object,
            then three sets of GT votes for up to three objects. If the point is only in one
            object's OBB, then the three GT votes are the same.
    """
    dataset = blender_object(data_dir,split=split)

    for data_idx in tqdm(range(len(dataset))):
        extract_blender_data_inner(data_idx,dataset,data_dir,split,num_point,type_whitelist,save_votes)
   

def extract_blender_data_multi(data_dir, split = 'train', num_point=20000, # Extracs data to .npz and .npy
    type_whitelist=DEFAULT_TYPE_WHITELIST, save_votes=True):
    """ Extract scene point clouds and 
    bounding boxes (centroids, box sizes, heading angles, semantic classes).
    Dumped point clouds and boxes are in upright depth coord.

    Args:
        root_dir: folder of the dataset
        split: training or testing
        save_votes: whether to compute and save Ground truth votes.

    Dumps:
        <id>_pc.npz of (N,3) where N is for number of subsampled points and 3 is
            for XYZ in upright depth coord
        <id>_bbox.npy of (K,8) where K is the number of objects, 10 is for
            centroids (cx,cy,cz), dimension (l,w,h), heanding_angle (ax,ay,az) and semantic_class
        <id>_votes.npz of (N,10) with 0/1 indicating whether the point belongs to an object,
            then three sets of GT votes for up to three objects. If the point is only in one
            object's OBB, then the three GT votes are the same.
    """
    dataset = blender_object(data_dir,split=split)

    from joblib import Parallel, delayed
    import multiprocessing

    num_cores = multiprocessing.cpu_count()
    test = Parallel(n_jobs=num_cores)(delayed(extract_blender_data_inner)(idx, dataset, data_dir, split, num_point, type_whitelist, save_votes) for idx in tqdm(range(len(dataset))))
    assert all(i == 0 for i in test), "Ein Unterprozess ist kaputt gegangen"

def extract_blender_data_inner(data_idx,dataset, data_dir, split, num_point, type_whitelist, save_votes):
    idx = dataset.index[data_idx]
    # Skip if XX/XXXX_votes.npz already exists
    if os.path.exists(os.path.join(data_dir, split, '{:02d}/{:04d}_votes.npz'.format(idx[0],idx[1]))): return 0

#    print('------------- ', data_idx+1, ' von ' , len(dataset))
    objects = dataset.get_label_objects(data_idx)

    # Skip scenes with 0 object or 0 objects out of type_whitelist
    if (len(objects)==0 or \
        len([obj for obj in objects if obj.classname in type_whitelist])==0):
        print(data_idx)
        return 1

    object_list = []
    for obj in objects:
        if obj.classname not in type_whitelist: continue
        obb = np.zeros(10)
        obb[0:3] = obj.centroid
        # Note that compared with that in data_viz, we do not time 2 to l,w.h
        # neither do we flip the heading angle
        obb[3:6] = np.array([obj.l,obj.w,obj.h])
        obb[6:9] = obj.heading_angle
        obb[9] = blender_utils.type2class[obj.classname]
        object_list.append(obb)
    if len(object_list)==0:
        obbs = np.zeros((0,10))
    else:
        obbs = np.vstack(object_list) # (K,10) K objects with 10 data entries

    pc_upright_depth = dataset.get_depth(data_idx)
    assert pc_upright_depth.shape[1] > 0, "Es gibt keine Datenpunkte in der Pointcloud"
    pc_upright_depth_subsampled = pc_util.random_sampling(pc_upright_depth, num_point)

    np.savez_compressed(os.path.join(data_dir, split, '{:02d}/{:04d}_pc.npz'.format(idx[0],idx[1])),
            pc=pc_upright_depth_subsampled)
    np.save(os.path.join(data_dir, split, '{:02d}/{:04d}_bbox.npy'.format(idx[0],idx[1])), obbs)
   
    if save_votes:
        N = pc_upright_depth_subsampled.shape[0]
        point_votes = np.zeros((N,10)) # 3 votes and 1 vote mask 
        point_vote_idx = np.zeros((N)).astype(np.int32) # in the range of [0,2]
        indices = np.arange(N)
        for obj in objects:
            if obj.classname not in type_whitelist: continue
            try:
                # Find all points in this object's OBB
                box3d_pts_3d = blender_utils.my_compute_box_3d(obj.centroid,np.array([obj.l,obj.w,obj.h]), obj.heading_angle)
                pc_in_box3d,inds = blender_utils.extract_pc_in_box3d(pc_upright_depth_subsampled, box3d_pts_3d)
                # Assign first dimension to indicate it is in an object box
                point_votes[inds,0] = 1
                # Add the votes (all 0 if the point is not in any object's OBB)
                votes = np.expand_dims(obj.centroid,0) - pc_in_box3d[:,0:3]
                sparse_inds = indices[inds] # turn dense True,False inds to sparse number-wise inds
                for i in range(len(sparse_inds)):
                    j = sparse_inds[i]
                    point_votes[j, int(point_vote_idx[j]*3+1):int((point_vote_idx[j]+1)*3+1)] = votes[i,:]
                    # Populate votes with the fisrt vote
                    if point_vote_idx[j] == 0:
                        point_votes[j,4:7] = votes[i,:]
                        point_votes[j,7:10] = votes[i,:]
                point_vote_idx[inds] = np.minimum(2, point_vote_idx[inds]+1)
            except:
                print('ERROR ----',  data_idx, obj.classname)
        np.savez_compressed(os.path.join(data_dir, split, '{:02d}/{:04d}_votes.npz'.format(idx[0],idx[1])),
            point_votes = point_votes)
    return 0

def get_box3d_dim_statistics(data_dir, split = 'train', # Computes the median box size for  BlenderDatasetConfig.type_mean_size
    type_whitelist=DEFAULT_TYPE_WHITELIST, save_path=None):
    """ Collect 3D bounding box statistics.
    Used for computing mean box sizes. """
    dataset = blender_object(data_dir,split=split)
    dimension_list = []
    type_list = []
    ry_list = []
    for idx in range(len(dataset)):
        print('------------- ', idx)
        calib = dataset.get_calibration(idx) # 3 by 4 matrix
        objects = dataset.get_label_objects(idx)
        for obj in objects:
            if obj.classname not in type_whitelist: continue
            heading_angle = obj.heading_angle
            dimension_list.append(np.array([obj.l,obj.w,obj.h])) 
            type_list.append(obj.classname) 
            ry_list.append(heading_angle)

    import pickle
    if save_path is not None:
        with open(save_path,'wb') as fp:
            pickle.dump(type_list, fp)
            pickle.dump(dimension_list, fp)
            pickle.dump(ry_list, fp)

    # Get average box size for different catgories
    box3d_pts = np.vstack(dimension_list)
    for class_type in sorted(set(type_list)):
        cnt = 0
        box3d_list = []
        for i in range(len(dimension_list)):
            if type_list[i]==class_type:
                cnt += 1
                box3d_list.append(dimension_list[i])
        median_box3d = np.median(box3d_list,0)
        print("\'%s\': np.array([%f,%f,%f])," % \
            (class_type, median_box3d[0]*2, median_box3d[1]*2, median_box3d[2]*2))

def extract_pointcloud_ply(data_dir):
    num_point = 80000
    for filename in tqdm(os.listdir(data_dir)):
        if filename.endswith(".ply"):
            name = filename.split('.')[:-1][0]
            ply = plyfile.PlyData.read(os.path.join(data_dir,filename)).elements[0].data
            pc = np.zeros((ply['x'].shape[0],3))
            print(name)
            pc[:,0] = ply['x']
            pc[:,1] = ply['y']
            pc[:,2] = ply['z']
            print("Before cut:", pc.shape)
            pc = ply_cut(pc,ply['z'])
            print("After cut:", pc.shape)
            pc_o3d = o3d.geometry.PointCloud()
            pc_o3d.points = o3d.utility.Vector3dVector(pc)
            voxel_down = pc_o3d.voxel_down_sample(voxel_size=0.002)
            cl,ind = voxel_down.remove_statistical_outlier(nb_neighbors=500,std_ratio=.02)
            pc = np.asarray(voxel_down.points)
            print("After Filtering:", pc.shape)
            assert pc.shape[1] > 0, "Es gibt keine Datenpunkte in der Pointcloud"
            pc_upright_depth_subsampled = pc_util.random_sampling(pc, num_point)
            np.savez_compressed(os.path.join(data_dir, '{:04d}_pc.npz'.format(int(name))), pc=pc_upright_depth_subsampled)
            pc_util.write_ply(pc_upright_depth_subsampled, os.path.join(data_dir, '{:04d}_new.ply'.format(int(name))))

def ply_cut(pc,z):
    mask = (z > -1)
    pc = pc[mask,:]
    return pc

if __name__=='__main__': # Run the different things implemented in this file.
    parser = argparse.ArgumentParser()
    parser.add_argument('--compute_median_size', action='store_true', help='Compute median 3D bounding box sizes for each class.')
    parser.add_argument('--gen_data', action='store_true', help='Generate dataset.')
    parser.add_argument('--gen_data_multi', action='store_true', help='Generate dataset with multiple processor cores.')
    parser.add_argument('--gen_ply_data', action='store_true', help='Generate dataset with data out of ply files.')
    parser.add_argument('--data_dir', default='/storage/data/blender_full/abc6/', help='Path to dataset.')
    args = parser.parse_args()

    if args.compute_median_size:
        get_box3d_dim_statistics(args.data_dir)
        exit()

    if args.gen_data:
        extract_blender_data(args.data_dir, split = 'train', save_votes = True)
        extract_blender_data(args.data_dir, split = 'test', save_votes = True)

    if args.gen_data_multi:
        import time
        extract_blender_data_multi(args.data_dir, split = 'train', save_votes = True)
        extract_blender_data_multi(args.data_dir, split = 'test', save_votes = True)
    
    if args.gen_ply_data:
        extract_pointcloud_ply(args.data_dir)
