# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

class BlenderDatasetConfig(object): # Should work
    def __init__(self):
        self.num_class = 10
        self.num_heading_bin = 12
        self.num_size_cluster = 10

        self.type2class={'ritzel':0, 'obj_1':1, 'obj_2':2, 'obj_3':3, 'obj_4':4, 'obj_5':5, 'obj_6':6, 'krones_1':7, 'krones_2':8, 'krones_3':9}
        self.class2type = {self.type2class[t]:t for t in self.type2class}
        self.type2onehotclass={'ritzel':0, 'obj_1':1, 'obj_2':2, 'obj_3':3, 'obj_4':4, 'obj_5':5, 'obj_6':6, 'krones_1':7, 'krones_2':8, 'krones_3':9}
        self.type_mean_size = {
                                'krones_1': np.array([2.544550,1.151976,1.964360]),
                                'krones_2': np.array([2.482387,1.151976,1.962702]),
                                'krones_3': np.array([2.942395,1.864898,1.118939]),
                                'obj_1': np.array([2.766266,2.607390,2.206797]),
                                'obj_2': np.array([3.439941,0.721894,0.045118]),
                                'obj_3': np.array([1.964326,1.609607,1.257092]),
                                'obj_4': np.array([2.667000,2.900947,1.334476]),
                                'obj_5': np.array([2.767263,2.967790,1.143836]),
                                'obj_6': np.array([3.448919,3.448919,0.300789]),
                                'ritzel': np.array([0.726274,0.728566,1.980000]),
                        }

        self.mean_size_arr = np.zeros((self.num_size_cluster, 3))
        for i in range(self.num_size_cluster):
            self.mean_size_arr[i,:] = self.type_mean_size[self.class2type[i]]

    def size2class(self, size, type_name):
        ''' Convert 3D box size (l,w,h) to size class and size residual '''
        size_class = self.type2class[type_name]
        size_residual = size - self.type_mean_size[type_name]
        return size_class, size_residual
    
    def class2size(self, pred_cls, residual):
        ''' Inverse function to size2class '''
        mean_size = self.type_mean_size[self.class2type[pred_cls]]
        return mean_size + residual
    
    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class
            [optinal] also small regression number from  
            class center angle to current angle.
           
            angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
            return is class of int32 of 0,1,...,N-1 and a number such that
                class*(2pi/N) + number = angle
        '''
        class_id = np.array((0,0,0))
        assert 0 <= angle.any() , "One angle smaller than 0"
        assert angle.any() <= 2*np.pi, "One angle bigger than 2*Pi"
        residual_angle = angle

        return class_id, residual_angle
    
    def class2angle(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class '''
        angle = residual
        assert 0 <= angle.any() , "One angle smaller than 0"
        assert angle.any() <= 2*np.pi, "One angle bigger than 2*Pi"
        return angle

    def param2obb(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle(heading_class, heading_residual)
        box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((9,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6:9] = np.dot(-1,heading_angle)
        return obb
