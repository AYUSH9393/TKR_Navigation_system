# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:53:58 2024

@author: admin
"""

import Trans3D
import numpy as np



def compute_rigid_body_transform(point_matrix, target_matrix):
    T_mat = Trans3D.affine_matrix_from_points(point_matrix, target_matrix, shear=False, scale=False, usesvd=False)
    return T_mat


def apply_affine_transform(points, T_mat):
    ndim = points.shape[0]
    num_point = points.shape[1]
    tmp = np.concatenate((points.T, np.ones([num_point, 1])), axis=1)
    tmp = np.dot(tmp, T_mat.T)
    transformed_points = tmp[:, :ndim]
    transformed_points = transformed_points.T
    return transformed_points  # 3 X number of points


# point matrix: ref in camera coordinates : 3XN
# target_matrix: ref in visrtual coordinates : 3XN
# tmp_points: pivot points in camera coordinates: 3XN
#T_points: pivot points in virtual refernce coordinates 
print("RigidBody")
#T_mat = compute_rigid_body_transform(point_matrix, target_matrix)
#T_points = apply_affine_transform(tmp_points, T_mat)
