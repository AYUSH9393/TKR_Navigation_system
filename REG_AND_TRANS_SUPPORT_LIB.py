# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:19:05 2024

@author: admin
"""


import itertools
import numpy as np
import matplotlib.pyplot as plt



def crossTimesMatrix(V):
    a = V.shape[0]
    b = V.shape[1]

    V_times = np.zeros(shape=(a, 3, b))

    V_times[0, 1, :] = -V[2, :]
    V_times[0, 2, :] = V[1, :]
    V_times[1, 0, :] = V[2, :]
    V_times[1, 2, :] = -V[0, :]

    V_times[2, 0, :] = -V[1, :]
    V_times[2, 1, :] = V[0, :]

    return V_times


def quat2rot(Q):
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
    R = np.zeros(shape=(3, 3))

    R[0][0] = q0*q0 + q1*q1 - q2*q2 - q3*q3
    R[0][1] = 2 * (q1*q2 - q0*q3)
    R[0][2] = 2 * (q1*q3 + q0*q2)

    R[1][0] = 2 * (q1*q2 + q0*q3)
    R[1][1] = q0*q0 - q1*q1 + q2*q2 - q3*q3
    R[1][2] = 2 * (q2*q3 - q0*q1)

    R[2][0] = 2 * (q1*q3 - q0*q2)
    R[2][1] = 2 * (q2*q3 + q0*q1)
    R[2][2] = q0*q0 - q1*q1 - q2*q2 + q3*q3

    return R


def estimateRigidTransform(x, y):
    if (x.shape[0] != 3) or (y.shape[0] != 3):
        raise Exception("Input point clouds must be a 3xN matrix.")
    if x.shape[1] != y.shape[1]:
        raise Exception("Input point clouds must be of the same size")
    if (x.shape[1] < 3) or (y.shape[1] < 3):
        raise Exception("At least 3 point matches are needed")

    pointCount = x.shape[1]

    x_centroid = (np.sum(x, axis=1)/pointCount).reshape(3, 1)
    y_centroid = (np.sum(y, axis=1)/pointCount).reshape(3, 1)

    x_centrized = x-x_centroid
    y_centrized = y-y_centroid

    R12 = y_centrized.T-x_centrized.T
    R21 = x_centrized-y_centrized
    R22_1 = y_centrized+x_centrized
    R22 = crossTimesMatrix(R22_1)

    B = np.zeros(shape=(4, 4))
    A = np.zeros(shape=(4, 4, pointCount))

    for i in range(pointCount):
        A[:, :, i][0][0] = 0
        A[:, :, i][0, 1:] = R12[i, :]
        A[:, :, i][1:, 0] = R21[:, i]
        A[:, :, i][1:, 1:] = R22[:, :, i]
        B = B+np.dot((A[:, :, i]).T, (A[:, :, i]))

    [U, s, v] = np.linalg.svd(B, full_matrices=True)
    S = np.diag(s)
    V = v.T

    quat = V[:, 3]

    rot = quat2rot(quat)


    T1 = np.array([[1, 0, 0, float(-y_centroid[0])], [0, 1, 0,
                                                      float(-y_centroid[1])], [0, 0, 1, float(-y_centroid[2])], [0, 0, 0, 1]])
    T2 = np.zeros(shape=(4, 4))
    T2[0:3, 0:3] = rot
    T2[3][3] = 1

    T3 = np.array([[1, 0, 0, float(x_centroid[0])], [0, 1, 0, float(
        x_centroid[1])], [0, 0, 1, float(x_centroid[2])], [0, 0, 0, 1]])

    T = np.dot(np.dot(T3, T2), T1)
    Eps = S[3][3]

    return T, Eps

def apply_affine_transform(points, T_mat):
    # points : 3X1
    ndim = points.shape[0]
    num_point = points.shape[1]
    tmp = np.concatenate((points.T, np.ones([num_point, 1])), axis=1)
    tmp = np.dot(tmp, T_mat.T)
    transformed_points = tmp[:, :ndim]
    transformed_points = transformed_points.T
    return transformed_points  # 3 X number of points

def compute_pivot_point(P1, P2):
    B = np.sum((np.square(P2) - np.square(P1)), axis=1)/2
    ans = np.linalg.lstsq(P2-P1, B, rcond=None)
    
    return ans[0]

def sphere_fit(point_cloud):
    """
    input
        point_cloud: xyz of the point clouds　numpy array
    output
        radius : radius of the sphere
        sphere_center : xyz of the sphere center
    """

    A_1 = np.zeros((3,3))
    #A_1 : 1st item of A
    v_1 = np.array([0.0,0.0,0.0])
    v_2 = 0.0
    v_3 = np.array([0.0,0.0,0.0])
    # mean of multiplier of point vector of the point_clouds
    # v_1, v_3 : vector, v_2 : scalar

    N = len(point_cloud)
    #N : number of the points

    """Calculation of the sum(sigma)"""
    for v in point_cloud:
        v_1 += v
        v_2 += np.dot(v, v)
        v_3 += np.dot(v, v) * v

        A_1 += np.dot(np.array([v]).T, np.array([v]))

    v_1 /= N
    v_2 /= N
    v_3 /= N
    A = 2 * (A_1 / N - np.dot(np.array([v_1]).T, np.array([v_1])))
    # formula ②
    b = v_3 - v_2 * v_1
    # formula ③
    sphere_center = np.dot(np.linalg.inv(A), b)
    #　formula ①
    radius = (sum(np.linalg.norm(np.array(point_cloud) - sphere_center, axis=1))
              /len(point_cloud))

    return(radius, sphere_center)

def Comput_Pivot_Point_FitSphere(pontcloud3d):
    # sphereFit(spX,spY,spZ):
    #   Assemble the A matrix
    spX = np.array(pontcloud3d[:,0])
    spY = np.array(pontcloud3d[:,1])
    spZ = np.array(pontcloud3d[:,2])
    A = np.zeros((len(spX),4))
    A[:,0] = spX*2
    A[:,1] = spY*2
    A[:,2] = spZ*2
    A[:,3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX),1))
    f[:,0] = (spX*spX) + (spY*spY) + (spZ*spZ)
    C, residules, rank, singval = np.linalg.lstsq(A,f,rcond=None)

    #   solve for the radius
    #t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    #radius = np.sqrt(t)

    return C


def get_pivot_point(PP,do_plot=False):
    #PP is N x 3 x p matrix of pivot tool
    #where p is the number of points in each point cloud in N clouds
    #for example aruco and QR have p=4
    pv_tool = np.squeeze(PP[0, :, :]).T
    p_pairs = np.array(
        list(itertools.combinations(np.arange(0, PP.shape[0]), 2)))  


    num_example = 10
    for k in range(20):
        P1 = []
        P2 = []
        rp = np.arange(np.size(p_pairs, axis=0))
        rp = np.random.permutation(rp)
        rp = rp[:num_example]

        rp_a = p_pairs[rp, 0]
        rp_b = p_pairs[rp, 1]

        P1 = np.array(np.squeeze(PP[rp_a[0], :, :]).T)
        P2 = np.array(np.squeeze(PP[rp_b[0], :, :]).T)

        for i in range(1, 10):

            P1 = np.concatenate(
                (P1, np.squeeze(PP[rp_a[i], :, :]).T), axis=0)
            P2 = np.concatenate(
                (P2, np.squeeze(PP[rp_b[i], :, :]).T), axis=0)
            
         
        X = compute_pivot_point(P1, P2)
        X = X.reshape(X.shape[0], 1)
        print(pv_tool.T.shape)
        [R, eps] = estimateRigidTransform(pv_tool.T, np.squeeze(PP[0, :, :]))


        tmp_pivot_points = np.array(apply_affine_transform(X, R)).T
        for j in range(1, PP.shape[0]):
            [R, eps] = estimateRigidTransform(
                pv_tool.T, np.squeeze(PP[j, :, :]))
            tmp_pivot_points = np.concatenate(
                (tmp_pivot_points, np.array(apply_affine_transform(X, R)).T), axis=0)


        if(k == 0):
            pivot_points = np.mean(tmp_pivot_points,axis=0).reshape(3,1).T
            print(pivot_points.shape)

        
        else:
            pivot_points = np.concatenate(
                (pivot_points, np.mean(tmp_pivot_points,axis=0).reshape(3,1).T))
            
    if do_plot == True:        
        fig=plt.figure()
        ax=plt.axes(projection='3d')
        ax.scatter3D(pv_tool[:,0],pv_tool[:,1],pv_tool[:,2])
        ax.scatter3D(pivot_points[:,0],pivot_points[:,1],pivot_points[:,2])
        plt.grid()
        plt.show()

    pv={}
    pv["pv_point"]=np.median(pivot_points,axis=0)
    pv["pv_tool"]=pv_tool
    print(pivot_points)
    return pv


'''    
PP = np.load('pv_corner_points.npy')  
pv = get_pivot_point(PP)
pv2 = Comput_Pivot_Point_FitSphere(np.mean(PP,axis=2))
'''

# where PP is N x 3 x 4 matrix of pivot tool 