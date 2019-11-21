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

        self.type2class={'ritzel':0, 'obj_1':1, 'obj_2':2, 'obj_3':3, 'obj_4':4, 'obj_5':5, 'obj_6':6, 'obj_7':7, 'obj_8':8, 'obj_9':9}
        self.class2type = {self.type2class[t]:t for t in self.type2class}
        self.type2onehotclass={'ritzel':0, 'obj_1':1, 'obj_2':2, 'obj_3':3, 'obj_4':4, 'obj_5':5, 'obj_6':6, 'obj_7':7, 'obj_8':8, 'obj_9':9}
        self.type_mean_size = {
                                'obj_1': np.array([2.680000,2.680000,0.490000]),
                                'obj_2': np.array([1.799441,1.800000,1.400000]),
                                'obj_3': np.array([0.555625,2.564100,0.560000]),
                                'obj_4': np.array([1.799441,1.800000,1.800000]),
                                'obj_5': np.array([0.840000,2.768000,2.000000]),
                                'obj_6': np.array([0.888940,1.840990,2.746188]),
                                'obj_7': np.array([2.055361,2.056000,1.760000]),
                                'obj_8': np.array([0.932688,2.190081,1.948131]),
                                'obj_9': np.array([2.340000,2.340000,0.520000]),
                                'ritzel': np.array([1.129025,1.132590,3.078000]),
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
