# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pickle
import itertools
from scipy.spatial.distance import pdist, cdist
#import pyrealsense2 as rs
# import cv2.aruco as aruco
from camera_calibrate import StereoCalibration
# from CAMERA_CAPTURE_MODULE_MAIN import ComeraCapture_RealSense, CameraCapture_USB
import time


class ArUco():
    def __init__(self):
        self.markersize = 4
        self.totalMarkers = 50      
        self.tool_vector =  None
        
    def set_tool_vector(self,tool_list):
        self.tool_vector = tool_list
        
    def get_corner_world_points(self,StereoObj, left_img, right_img):
        tool_ret, matched_left_corner_points, matched_right_corner_points = self.get_matched_corner_points_pairs(left_img, right_img, False)
        wp_list = [] 
        
        #print('------right corner points-------')
        #print(matched_right_corner_points)
        for i in range(len(tool_ret)):
            if tool_ret[i]==True:
                wp = StereoObj.get_world_points(matched_left_corner_points[i], matched_right_corner_points[i])
                #print('wp',wp)
                wp_list.append(wp)
            else:
                wp_list.append(None)       
        return tool_ret, wp_list
    
    def get_center_world_points(self,StereoObj, left_img, right_img):
        tool_ret, matched_left_corner_points, matched_right_corner_points = self.get_matched_corner_points_pairs(left_img, right_img, True)
        wp_list = [] 
        for i in range(len(tool_ret)):
            if tool_ret[i]==True:
                left_cnt_pnts = np.expand_dims(np.mean(matched_left_corner_points[i], axis=0), axis=0)
                right_cnt_pnts = np.expand_dims(np.mean(matched_right_corner_points[i], axis=0), axis=0)
                wp = StereoObj.get_world_points(left_cnt_pnts, right_cnt_pnts)
                #print('wp',wp)
                wp_list.append(wp)
            else:
                wp_list.append(None)       
        return tool_ret, wp_list
    
    def get_corner_world_points_v2(self,StereoObj, left_img, right_img):
        tool_ret, matched_left_corner_points, matched_right_corner_points = self.get_matched_corner_points_pairs(left_img, right_img)
        wp_list = [] 
        for i in range(len(tool_ret)):
            if tool_ret[i]==True:
                wp = StereoObj.get_world_points(matched_left_corner_points[i], matched_right_corner_points[i])
                #print('wp',wp)
                wp_list.append(wp)
            else:
                wp_list.append(None)       
        return tool_ret, wp_list
                
    def findArucoMarkers(self, img, draw=True):
        if img is None or img.size == 0:
            print("[ERROR] Image is empty or not loaded properly.")
            return None, None

        # Get dictionary
        key = getattr(cv2.aruco, f'DICT_{self.markersize}X{self.markersize}_{self.totalMarkers}')

        # Detector parameters
        try:
            # OpenCV ≥ 4.7 (new API)
            aruco_dict = cv2.aruco.getPredefinedDictionary(key)
            aruco_params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
            bboxes, ids, rejected = detector.detectMarkers(img)
        except AttributeError:
            # Older API
            aruco_dict = cv2.aruco.Dictionary_get(key)
            aruco_params = cv2.aruco.DetectorParameters_create()
            bboxes, ids, rejected = cv2.aruco.detectMarkers(img, aruco_dict, parameters=aruco_params)

        if ids is None or len(ids) == 0:
            return None, None

        if draw:
            cv2.aruco.drawDetectedMarkers(img, bboxes, ids)

        return np.array(bboxes), np.array(ids)
    
    def get_matched_corner_points_pairs(self, left_img, right_img, draw):
        num_tools = len(self.tool_vector)
        matched_left_corner_points = np.empty((num_tools, 4, 2))
        matched_right_corner_points = np.empty((num_tools, 4, 2))
        tool_ret = [False] * num_tools

        corner_left, ids_left = self.findArucoMarkers(left_img, draw)
        corner_right, ids_right = self.findArucoMarkers(right_img, draw)

        # Ensure arrays for matching
        if ids_left is None or ids_right is None:
            return tool_ret, matched_left_corner_points, matched_right_corner_points

        for i in range(num_tools):
            # Compare safely
            idx_l = np.where(ids_left.flatten() == self.tool_vector[i])[0]
            idx_r = np.where(ids_right.flatten() == self.tool_vector[i])[0]

            if idx_l.size == 1 and idx_r.size == 1:
                tool_ret[i] = True
                matched_left_corner_points[i, :, :] = corner_left[idx_l[0], :, :]
                matched_right_corner_points[i, :, :] = corner_right[idx_r[0], :, :]

        return tool_ret, matched_left_corner_points, matched_right_corner_points
                    
class SterioParameter():
    def __init__(self,stereo_para_file):
        with open(stereo_para_file, 'rb') as file_handle:
            print('stereo parameter object loaded...')
            self.cal = pickle.load(file_handle)
            
        self.wL= 1280 #3840
        self.hL= 720 #1080chnge
        self.New_M1, tmp = cv2.getOptimalNewCameraMatrix(self.cal.camera_model['M1'],self.cal.camera_model['dist1'],(self.wL,self.hL),1,(self.wL,self.hL))
        self.New_M2, tmp = cv2.getOptimalNewCameraMatrix(self.cal.camera_model['M2'],self.cal.camera_model['dist2'],(self.wL,self.hL),1,(self.wL,self.hL))
                      
        self.projMat_L = self.cal.camera_model['M1'] @ cv2.hconcat([np.eye(3), np.zeros((3,1))]) # Cam1 is the origin
        self.projMat_R= self.cal.camera_model['M2'] @ cv2.hconcat([self.cal.camera_model['R'], self.cal.camera_model['T']]) # R, T from stereoCalibrate
        
        self.sr_rotation1, self.sr_rotation2, self.sr_pose1, self.sr_pose2 = \
        cv2.stereoRectify(cameraMatrix1 = self.cal.M1, 
                          distCoeffs1 = self.cal.d1,
                          cameraMatrix2 = self.cal.M2, 
                          distCoeffs2 = self.cal.d2, 
                          imageSize = (self.wL,self.hL),
                          R = self.cal.camera_model['R'], 
                          T = self.cal.camera_model['T']                                                   
                          )[0:4]
        print("R",self.cal.camera_model['R'])
        print("T",self.cal.camera_model['T'])
        #print("cameraMatrix1",self.cal.M1)
        self.scale_factor = 25 #change 31mm 24.7 (Square size of checkerboard in mm)
        self.R = self.cal.camera_model['R']
        self.T = self.cal.camera_model['T']

    def R_and_T(self):      
        return self.R, self.T
     
    def get_world_points(self, L_key_pnts, R_key_pnts):       
        points3d = np.empty((0,0))
        L_key_pnts = L_key_pnts.reshape(L_key_pnts.shape[0],1,2)
        R_key_pnts = R_key_pnts.reshape(R_key_pnts.shape[0],1,2)
        L_key_pnts = cv2.undistortPoints(L_key_pnts,self.cal.M1,self.cal.d1,None, self.cal.camera_model['M1'])
        R_key_pnts = cv2.undistortPoints(R_key_pnts,self.cal.M2,self.cal.d2,None, self.cal.camera_model['M2'])
        
   
        points4d = cv2.triangulatePoints(self.projMat_L, self.projMat_R, L_key_pnts, R_key_pnts)        
        #points4d = cv2.triangulatePoints(self.projMat_L, self.projMat_R, L_key_pnts.reshape(L_key_pnts.shape[0],1,2), R_key_pnts.reshape(R_key_pnts.shape[0],1,2))            
        points3d = (points4d[:3, :]/points4d[3, :]).T
        points3d = points3d*self.scale_factor  
            
        return points3d
        
        
    def get_world_points_product(self, L_key_pnts, R_key_pnts):       
        L_key_pnts = L_key_pnts.reshape(L_key_pnts.shape[0],1,2)
        R_key_pnts = R_key_pnts.reshape(R_key_pnts.shape[0],1,2)
        L_key_pnts = cv2.undistortPoints(L_key_pnts,self.cal.M1,self.cal.d1,None, self.cal.camera_model['M1'])
        R_key_pnts = cv2.undistortPoints(R_key_pnts,self.cal.M2,self.cal.d2,None, self.cal.camera_model['M2'])
        

        comb_idx = np.asarray(list(itertools.product(range(L_key_pnts.shape[0]),range(L_key_pnts.shape[0]))))
        #print(comb_idx)
        L_key_pnts = L_key_pnts[comb_idx[:,0],:,:]
        R_key_pnts = R_key_pnts[comb_idx[:,1],:,:]
        
        #print(L_key_pnts.shape)
        #print(R_key_pnts.shape)
        
        points4d = cv2.triangulatePoints(self.projMat_L, self.projMat_R, L_key_pnts, R_key_pnts)
        #points4d = cv2.triangulatePoints(self.projMat_L, self.projMat_R, L_key_pnts.reshape(L_key_pnts.shape[0],1,2), R_key_pnts.reshape(R_key_pnts.shape[0],1,2))
        points3d = (points4d[:3, :]/points4d[3, :]).T
        points3d = points3d*self.scale_factor 
        return points3d
 
