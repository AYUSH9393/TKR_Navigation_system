
import numpy as np
import REG_AND_TRANS_SUPPORT_LIB
import math





# INPUT: source and target is 4x3
# OUTPUT: Estimated transform T
def Comput_Rigid_Body_Transform(source, target):
    source = np.asarray(source, dtype=float).T
    target = np.asarray(target, dtype=float).T    
    T, Eps = REG_AND_TRANS_SUPPORT_LIB.estimateRigidTransform(target,source)   
    return T
    
# def Comput_Rigid_Body_Transform(source, target):
#     source = np.asarray(source, dtype=float).T
#     target = np.asarray(target, dtype=float).T
    
#     # Ensure estimateRigidTransform returns exactly two values
#     result = REG_AND_TRANS_SUPPORT_LIB.estimateRigidTransform(target, source)
    
#     if isinstance(result, tuple) and len(result) == 2:
#         T, Eps = result
#     else:
#         T = result  # Assume only the transformation matrix is returned
#         Eps = None  # Set Eps to None or a default value
    
#     return T, Eps

#INPUT pnts = N X 3
#OUTPUT  transfomed point Nx3
def apply_tranform_to_points(pnts, T):
    if len(pnts.shape)==1:
        pnts = np.expand_dims(pnts,axis=0)        
    pnts = np.asarray(pnts, dtype=float).T
    pnts = REG_AND_TRANS_SUPPORT_LIB.apply_affine_transform(pnts, T)
    #print(pnts)
    return pnts.T


class SQ_MARKER_PIVOT_OBJ:
    #pivot_def = 5X3 p
    def __init__(self, pivot_def):
        self.pivot_def = pivot_def
        #print("pivot_def: \n",pivot_def)
    def set_ref_points(self,ref_points):
        self.ref_points = ref_points
    def get_pivot_point_CamCoordSys(self,pnts):
        T = Comput_Rigid_Body_Transform(source=self.pivot_def[:4,:], target=pnts) # 4x3
        P = apply_tranform_to_points(self.pivot_def, T) # 4x3
        return P # pivot point
    def get_pivot_point_RefCoordSys(self, pnts):
        T = Comput_Rigid_Body_Transform(source=self.pivot_def[:4,:], target=pnts)
        P = apply_tranform_to_points(self.pivot_def, T)
        T = Comput_Rigid_Body_Transform(source=P[:4,:], target=self.ref_points)
        P = apply_tranform_to_points(P, T)
        return P[4,:] # pivot point


# marker size of SQ marker
def get_vertual_SQ_coordinates(marker_size):
    
    vm = [[-1, 1, 0],
          [ 1, 1, 0],
          [ 1,-1, 0],
          [-1,-1, 0]]   
    vm = np.asarray(vm, dtype=float)
    return vm*(marker_size/2.0)

def GET_ARUCO_VREF_TRANSFORM(four_corner_points, marker_size=None):
    if marker_size is None:
        aruco_center = np.mean(four_corner_points, axis=0)
        ms1 = np.linalg.norm(four_corner_points[0]-four_corner_points[1])
        ms2 = np.linalg.norm(four_corner_points[1]-four_corner_points[2])
        ms3 = np.linalg.norm(four_corner_points[2]-four_corner_points[3])
        ms4 = np.linalg.norm(four_corner_points[3]-four_corner_points[0])
        marker_size = (ms1+ms2+ms3+ms4)/4
        print("marker_size",marker_size)
    virtual_sq_marker = get_vertual_SQ_coordinates(marker_size)
    T  = Comput_Rigid_Body_Transform(source=four_corner_points, target = virtual_sq_marker)
    return T

def Comput_Aruco_Pivot_Point(N_sq_conrner_points):
    
    # N_sq_conrner_points Nx4x3
    #fit a given set of 3D points (x, y, z) to a sphere  
    
    pnts = np.asarray(np.mean(N_sq_conrner_points,axis=1)) # get center of aruco
    row_num = pnts.shape[0]
    A = np.ones((row_num, 4))
    A[:,0:3] = pnts
    
    # construct vector f
    f = np.sum(np.multiply(pnts, pnts), axis=1)
    
    sol, residules, rank, singval = np.linalg.lstsq(A,f,rcond=None)
    
    # solve the radius
    radius = math.sqrt((sol[0]*sol[0]/4.0)+(sol[1]*sol[1]/4.0)+(sol[2]*sol[2]/4.0)+sol[3])
    
    print("radius:",radius)
    
    pv_point = np.asarray([sol[0]/2.0, sol[1]/2.0, sol[2]/2.0])
    print(pv_point)
    
    vref_pivot_points = np.zeros((N_sq_conrner_points.shape[0],3))
    for i in range(N_sq_conrner_points.shape[0]):
        T = GET_ARUCO_VREF_TRANSFORM(N_sq_conrner_points[i,:,:])
        vref_pivot_points[i,:] = apply_tranform_to_points(np.tile(pv_point,[2,1]), T)[0]
        print(vref_pivot_points[i,:])
    
    pv_point = np.median(vref_pivot_points, axis=0)
    return pv_point


# sample_data = np.array([
#     [
#         [0, 0, 0],  # Corner 1
#         [1, 0, 0],  # Corner 2
#         [1, 1, 0],  # Corner 3
#         [0, 1, 0]   # Corner 4
#     ],
#     [
#         [0.1, 0.1, 0.1],
#         [1.1, 0.1, 0.1],
#         [1.1, 1.1, 0.1],
#         [0.1, 1.1, 0.1]
#     ],
#     [
#         [0.2, 0.2, 0.2],
#         [1.2, 0.2, 0.2],
#         [1.2, 1.2, 0.2],
#         [0.2, 1.2, 0.2]
#     ],
#     [
#         [0.3, 0.3, 0.3],
#         [1.3, 0.3, 0.3],
#         [1.3, 1.3, 0.3],
#         [0.3, 1.3, 0.3]
#     ],
#     [
#         [0.4, 0.4, 0.4],
#         [1.4, 0.4, 0.4],
#         [1.4, 1.4, 0.4],
#         [0.4, 1.4, 0.4]
#     ]
# ])
      
# cp = np.load('pv_corner_points_v2.npy')   
# pv_point = Comput_Aruco_Pivot_Point(np.transpose(cp,[0, 2, 1]))   
# print("Computed Pivot Point:", pv_point)

# pv_point = Comput_Aruco_Pivot_Point(sample_data)
# print("Computed Pivot Point:", pv_point)