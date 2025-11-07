import numpy as np
import json
import math
import cv2
import os
import itertools
import matplotlib.pyplot as plt
import scipy.io as spio
import time
from threading import Thread
import socket
from enum import Enum
import rigid_body_transform_info
import Trans3D
import configparser
from main1 import data_1
from CAMERA_CAPTURE_MODULE_MAIN import CameraCapture_Pi as CameraCapture_USB
from collections import deque
import SQ_MARKER_NAVIGATION_SUPPORT_LIB
from SQ_MARKER_NAVIGATION_SUPPORT_LIB import SQ_MARKER_PIVOT_OBJ
import REG_AND_TRANS_SUPPORT_LIB
import Planes_genrate_LIB
#import pyvista as pv
from sympy import Plane, Point, Point3D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pygame
import socket
import struct
import tkinter as tk
from tkinter import messagebox
import tkinter.font as tkFont
#Paddle Markers : 0-Pointer, 1- Verification, 2-tibia, 3- femur
from PIL import Image, ImageTk
import threading
import sys
#from app import app_state, update_frame, update_results, get_command_state, reset_capture_flag
import threading
import xml.etree.ElementTree as ET
import os
#nik
#socket related Code & variables
#---------------------------------
# Global references
camera_on_img = None
camera_off_img = None
camera_icon_label = None




pointCount = 0
messageID = ''
corner_point = []
zeroArr = np.zeros((4,3))
config = configparser.ConfigParser()
config.read('config.ini')


import threading
import sys

# Start Flask server in background thread BEFORE main code runs
try:
    from flask_server import (
        run_flask_server, 
        update_frame, 
        update_results, 
        get_command_state, 
        reset_capture_flag,
        update_point_count,
        reset_reset_flag,
        update_current_point,
        update_marker_status 
    )
    
    # Start Flask in background thread
    flask_thread = threading.Thread(target=run_flask_server, daemon=True)
    flask_thread.start()
    print("[Main] Flask server started in background")
    
    # Give Flask time to initialize
    import time
    time.sleep(2)
    
    WEB_MODE_ENABLED = True
except ImportError as e:
    print(f"[Warning] Flask not available: {e}")
    print("[Main] Running in keyboard-only mode")
    WEB_MODE_ENABLED = False
    
    # Define dummy functions if Flask not available
    def update_frame(frame): pass
    def update_results(*args, **kwargs): pass
    def get_command_state(): return {'selected_object': 'femur', 'ct_side': None, 'capture_point': False,'reset_triggered': False, 'points_captured': {'femur': 0, 'tibia': 0}}
    def reset_capture_flag(): pass
    def update_point_count(obj, count): pass
    def reset_reset_flag(): pass 
    def update_current_point(message): pass
    def update_marker_status(calibrated, markers_data): pass 


femurFirstFrame = None
OnlyAtFirstTime = True
is_taking_Hip_Center = True
removedFirstValue = False

femur_list = []
tibia_list = []
relative_femur_points_list = []
relative_tibia_points_list = []
hip_center_fetcher_list = []
hip_center_fetcher_print_list = []
ac_Fetcher_List = []

selectedObject = None
CT_side = None

tibiaCount = None
femurCount = None

T_ref_tib = None
T_ref_fem = None

corner_point = None
center_points = None
zero_arr = np.zeros((4, 3))
zeroArr = zero_arr

prev_vp_normal = None
prev_center = None

tibia_point_names = [
    "Tibia Knee Center",
    "Tibia Medial Condyle",
    "Tibia Lateral Condyle",
    "Tuberosity",
    "PCL",
    "Medial Malleolus",
    "Lateral Malleolus",
    "Ankle Center Complete"
]


#New_start--------------------
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

def compute_pivot_point(P1, P2):
    B = np.sum((np.square(P2) - np.square(P1)), axis=1)/2
    ans = np.linalg.lstsq(P2-P1, B, rcond=None)
    return ans[0]

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


    T1 = np.array([
    [1, 0, 0, float(-y_centroid[0, 0])], 
    [0, 1, 0, float(-y_centroid[1, 0])], 
    [0, 0, 1, float(-y_centroid[2, 0])], 
    [0, 0, 0, 1]
])
    T2 = np.zeros(shape=(4, 4))
    T2[0:3, 0:3] = rot
    T2[3][3] = 1

    T3 = np.array([
    [1, 0, 0, float(x_centroid[0, 0])], 
    [0, 1, 0, float(x_centroid[1, 0])], 
    [0, 0, 1, float(x_centroid[2, 0])], 
    [0, 0, 0, 1]
])

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

def get_pivot_point(PP):
   #pv_tool = PP.T
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

    pv={}
    pv["pv_point"]=np.median(pivot_points,axis=0)
    pv["pv_tool"]=pv_tool
    print(pivot_points)
    print("PVTool : ",pv["pv_tool"])
    return pv
#End------------------------------    

def getAverageOfFourCorners(fourCorners):
    '''
        It accepts 3x4 np.array for finding center
        and it returns x,y,z tuple
    '''
    result = np.array([0.0,0.0,0.0])
    for i in range(3):
        result[i] = np.average(fourCorners[i])
    result = tuple(result)
    return result


#corner arrays for Virtual
topleft = np.array([-30.0,30.0,0])
topright = np.array([30.0,30.0,0])
bottomright = np.array([30.0,-30.0,0])
bottomleft = np.array([-30.0,-30.0,0])
target_matrix = np.array([topleft,topright,bottomright,bottomleft])

T_mat = target_matrix
T_mat_femur = target_matrix
point_matrix = target_matrix

T_mat = target_matrix

def getTheIP():
    host = socket.gethostname()
    ipAddress = socket.gethostbyname(host)
    print("Got the IP")
    print(ipAddress)
    return ipAddress

def getRelativeToMarker(point_matrix, ref_marker):
                         #vp            tibia/femur   
    '''
    FOR REF MARKERS
    Function to apply Rigid transformations
    to defined point_matrix ( ref marker Corners 4x3 which is converted 3x4 in this function)
    and Returns 4x3 materix  -16.4564 -103.8722  648.2049]
    '''
    point_matrix = point_matrix.T #Transposing from 4x3 to 3x4
    ref_marker = ref_marker.T
    #print(target_matrix)
    #print(point_matrix)
    #getting T_mat
    T_mat = rigid_body_transform_info.compute_rigid_body_transform(ref_marker, target_matrix)
    #applying affine transform
    T_points = rigid_body_transform_info.apply_affine_transform(point_matrix,T_mat)
    T_points = T_points.T
    return T_points

def findAnkleCenter(A, B):
    alpha = 0.44  # 44% away from A
    M = (1 - alpha) * A + alpha * B
    print("Middle Point:", M)
    return M

ac_Fetcher_List = []
# Initialize these GLOBALLY (outside the function)
prev_button_a = False
last_button_a_trigger = 0

def tibia_pointtaker_o():
    global ac_Fetcher_List
    global ax
    global pivot_ref
    global prev_button_a  # Access the global variables
    global last_button_a_trigger

    pointer_corners = corner_point[0]  # 4x3 array of pointer marker's corners
    try:
        # Compute pivot point in camera coordinates
        pivot_cam = pivot_point_obj.get_pivot_point_CamCoordSys(pointer_corners)
        # Transform to reference coordinate system
        #pivot_ref = placeAPointer()
        pivot_ref = SQ_MARKER_NAVIGATION_SUPPORT_LIB.apply_tranform_to_points(pivot_cam, T_ref_tib)
        pivot_ref = pivot_ref[4,:]

        # Check if 'c' key is prescsed and the list has less than 2 values
        if not (corner_point[0] == zeroArr).all() and not (corner_point[2] == zeroArr).all():
            #key = cv2.waitKey(1) & 0xFF 
            # Inside your main loop where you check button presses
            #pygame.event.pump()  # Poll joystick events
            #current_time = pygame.time.get_ticks()

            # Get current button state
            #button_a_current = joystick.get_button(0)

            # Detect rising edge (transition from not pressed to pressed)
            #button_a_pressed = button_a_current and not prev_button_a
            #prev_button_a = button_a_current  

            # Check trigger conditions
            #key_pressed = (key == ord('c'))
            #button_allowed = button_a_pressed and (current_time - last_button_a_trigger >= 1000)
            
            #if (button_allowed) and len(tibia_list) <= 6 and CT_side_selected:
            #key = cv2.waitKey(1) & 0xFF
            #print("tibia_pointer")
            #if key == ord('c') and len(tibia_list) <= 6:
                # Update cooldown if triggered by button
                #if button_a_pressed:
                    #last_button_a_trigger = current_time

                if len(tibia_list) == 0:      #...................KC
                    print("KC")
                    tibia_list.append(pivot_ref)
                    # Get relative position
                    relative_pos = get_relative_position(corner_point[2], pivot_ref)
                    relative_tibia_points_list.append(relative_pos)
                    #tibia_list.append(center_points[0])
                    print(f"pivot_ref: {pivot_ref}")
                    #print("corner_point[0]: ", corner_point[0])
                    #draw_a_plot_pointerFourVal(pivot_ref,corner_point[0])
                    print(f"Stored pivot_ref: {pivot_ref}")
                    print(f"Current list: {tibia_list}")
                elif len(tibia_list) == 1:  #..................TMC
                    print("TMC")
                    tibia_list.append(pivot_ref)
                    # Get relative position
                    relative_pos = get_relative_position(corner_point[2], pivot_ref)
                    relative_tibia_points_list.append(relative_pos)
                    #tibia_list.append(center_points[0])
                    
                    #draw_a_plot_pointerFourVal(pivot_ref,corner_point[0])
                    print(f"Stored pivot_ref: {pivot_ref}")
                    print(f"Current list: {tibia_list}")
                elif len(tibia_list) == 2:  #..................TLC
                    print("TLC")
                    tibia_list.append(pivot_ref)
                    # Get relative position
                    relative_pos = get_relative_position(corner_point[2], pivot_ref)
                    relative_tibia_points_list.append(relative_pos)
                    #draw_a_plot_pointerFourVal(pivot_ref,corner_point[0])
                    #tibia_list.append(center_points[0])
                    print(f"Stored pivot_ref: {pivot_ref}")
                    print(f"Current list: {tibia_list}")
                elif len(tibia_list) == 3:
                    print("tuberosity")
                    tibia_list.append(pivot_ref)
                    relative_pos = get_relative_position(corner_point[2],pivot_ref)
                    relative_tibia_points_list.append(relative_pos)
                    print(f"Stored pivot_ref: {pivot_ref}")
                    print(f"Current list: {tibia_list}")
                elif len(tibia_list) == 4:
                    print("PCL")
                    tibia_list.append(pivot_ref)
                    relative_pos = get_relative_position(corner_point[2],pivot_ref)
                    relative_tibia_points_list.append(relative_pos)
                    print(f"Stored pivot_ref: {pivot_ref}")
                    print(f"Current list: {tibia_list}")
                else:                       #..................AC
                    print("Storing AC calc points")
                    ac_Fetcher_List.append(pivot_ref)
                    #draw_a_plot_pointerFourVal(pivot_ref,corner_point[0])
                    print(ac_Fetcher_List)
                    if(len(ac_Fetcher_List) == 2):
                        print("AC Finder Points taken")
                        print(ac_Fetcher_List)
                        Dist_medial_lateral = np.linalg.norm(ac_Fetcher_List[0] - ac_Fetcher_List[1])
                        print("Distance medial-lateral: ", Dist_medial_lateral)
                        tibia_list.append(findAnkleCenter(ac_Fetcher_List[0], ac_Fetcher_List[1]))
                        # Get relative position
                        relative_pos = get_relative_position(corner_point[2], pivot_ref)
                        relative_tibia_points_list.append(relative_pos)
                        ac_Fetcher_List = []
                        print("AC Found")
                        print(tibia_list)
                
    except Exception as e:
        print(f"Error computing pivot point: {e.with_traceback()}")

def capture_average_pivot_point(T_ref, duration=1.0, fps_estimate=30):
    """
    Capture multiple pivot points over 'duration' seconds
    and return the average point for better accuracy.
    """
    global pivot_point_obj, corner_point  # REMOVED T_ref_tib from globals

    pointer_corners = corner_point[0]
    collected_points = []

    start_time = time.time()
    while (time.time() - start_time) < duration:
        try:
            # Compute pivot for current frame
            pivot_cam = pivot_point_obj.get_pivot_point_CamCoordSys(pointer_corners)

            # Check if T_ref is valid
            if T_ref is None:
                print("[Error] Reference Transform (T_ref) is None in capture_average_pivot_point")
                return None # Fail fast if transform isn't set

            # USE THE PASSED T_ref ARGUMENT
            pivot_ref = SQ_MARKER_NAVIGATION_SUPPORT_LIB.apply_tranform_to_points(pivot_cam, T_ref)
            pivot_ref = pivot_ref[4, :]
            collected_points.append(pivot_ref)
        except Exception as e:
            print(f"[Warning] Frame skipped during avg capture: {e}")
            # Don't continue, wait for sleep

        # Small delay for ~30 FPS
        time.sleep(1.0 / fps_estimate)
    
    if not collected_points:
        print("[Error] No valid pivot points collected during averaging!")
        return None

    avg_point = np.median(collected_points, axis=0)
    print(f"[Capture] Averaged {len(collected_points)} frames over {duration}s -> {avg_point}")
    return avg_point

def update_next_tibia_point():
    """Update the UI with the next tibia point to capture."""
    total_tibia_points = len(tibia_list)
    ac_count = len(ac_Fetcher_List)

    if total_tibia_points == 0:
        update_current_point("Tibia Knee Center")
    elif total_tibia_points == 1:
        update_current_point("Tibia Medial Condyle")
    elif total_tibia_points == 2:
        update_current_point("Tibia Lateral Condyle")
    elif total_tibia_points == 3:
        update_current_point("Tuberosity")
    elif total_tibia_points == 4:
        update_current_point("PCL")
    elif total_tibia_points == 5 and ac_count == 0:
        update_current_point(" Medial Malleolus")
    elif total_tibia_points == 5 and ac_count == 1:
        update_current_point("Lateral Malleolus")
    elif total_tibia_points >= 6:
        update_current_point("Tibia Points Taken")

class SimpleKalman3D:
    def __init__(self, process_var=1e-3, measure_var=1e-1):
        # state: position only (x,y,z)
        self.x = None         # state (3,)
        self.P = None         # covariance (3x3)
        self.Q = np.eye(3) * process_var
        self.R = np.eye(3) * measure_var

    def initialize(self, init_pos):
        self.x = np.array(init_pos, dtype=float)
        self.P = np.eye(3) * 1.0

    def update(self, measurement):
        z = np.array(measurement, dtype=float)
        if self.x is None:
            self.initialize(z)
            return self.x.copy()

        # Predict (identity dynamics)
        x_pred = self.x
        P_pred = self.P + self.Q

        # Kalman gain
        S = P_pred + self.R
        K = P_pred.dot(np.linalg.inv(S))

        # Update
        self.x = x_pred + K.dot(z - x_pred)
        self.P = (np.eye(3) - K).dot(P_pred)
        return self.x.copy()


def median_of_samples(samples):
    arr = np.array(samples)   # N x 3
    return np.median(arr, axis=0)


def mad_mask(samples, thresh=3.5):
    """
    Return boolean mask of samples that are inliers based on MAD.
    samples: N x 3
    thresh: threshold in MAD units (default 3.5)
    """
    arr = np.array(samples)
    med = np.median(arr, axis=0)
    dev = np.abs(arr - med)
    mad = np.median(dev, axis=0)
    # Prevent divide-by-zero
    mad_safe = np.where(mad == 0, 1e-6, mad)
    # Convert each sample to a maximum scaled dev across axes
    scaled_dev = np.max(dev / mad_safe, axis=1)
    return scaled_dev <= thresh


def capture_median_pivot_point(duration=1.0, fps_estimate=30, use_kalman=False, 
                               mad_thresh=3.5, min_frames=5, max_frames=120):
    """
    Collect pivot points for `duration` seconds (approx), compute a median pivot
    after rejecting outliers via MAD. Optionally apply a 3D Kalman smoothing.

    Returns:
        avg_point (3,) numpy array or None on failure.
    """
    global pivot_point_obj, corner_point, T_ref_tib

    collected = []
    start = time.time()
    frame_interval = 1.0 / fps_estimate
    timeout = duration

    # Allow a maximum number of frames to avoid endless loops
    max_frames_allowed = max_frames

    while (time.time() - start) < timeout and len(collected) < max_frames_allowed:
        try:
            pointer_corners = corner_point[0]
            if pointer_corners is None:
                time.sleep(frame_interval)
                continue

            pivot_cam = pivot_point_obj.get_pivot_point_CamCoordSys(pointer_corners)
            pivot_ref = SQ_MARKER_NAVIGATION_SUPPORT_LIB.apply_tranform_to_points(pivot_cam, T_ref_tib)
            pivot_ref = pivot_ref[4, :]    # ensure shape (3,)
            collected.append(np.array(pivot_ref, dtype=float))
        except Exception as e:
            # skip bad frames
            # print(f"[capture_median] frame skip: {e}")
            pass

        time.sleep(frame_interval)

    if len(collected) < min_frames:
        print(f"[capture_median] Not enough frames collected: {len(collected)}")
        return None

    # 1) MAD outlier rejection
    mask = mad_mask(collected, thresh=mad_thresh)
    inliers = np.array(collected)[mask]

    # If all removed, fall back to median of all
    if inliers.shape[0] < min_frames:
        print(f"[capture_median] Too many outliers removed ({np.sum(~mask)}). Falling back to median of all frames.")
        result = median_of_samples(collected)
    else:
        result = np.median(inliers, axis=0)

    # Optional smoothing
    if use_kalman:
        kf = SimpleKalman3D(process_var=1e-4, measure_var=1e-2)
        # feed each inlier sequentially to kalman and get final state
        for sample in (inliers if inliers.shape[0] > 0 else np.array(collected)):
            kf.update(sample)
        result = kf.x

    print(f"[capture_median] Collected {len(collected)} frames, inliers {inliers.shape[0] if 'inliers' in locals() else len(collected)} -> pivot {result}")
    return result

def tibia_pointtaker():
    """
    Captures tibia points ONLY when explicitly triggered by keyboard or web button.
    Does NOT auto-capture when markers are detected.
    """
    global ac_Fetcher_List
    global tibia_list, relative_tibia_points_list
    global pivot_ref
    global T_ref_tib
    global corner_point, zeroArr

    # CRITICAL: Only proceed if markers are detected AND capture was triggered
    # This function should ONLY be called when user presses 'c' key or web capture button
    
    pointer_corners = corner_point[0]  # 4x3 array of pointer marker's corners
    
    try:
        # Check if both pointer and tibia markers are detected
        if (corner_point[0] == zeroArr).all() or (corner_point[2] == zeroArr).all():
            print("[Warning] Pointer or Tibia marker not detected")
            return False
        
        # Compute pivot point in camera coordinates
        #pivot_cam = pivot_point_obj.get_pivot_point_CamCoordSys(pointer_corners)
        
        # Transform to reference coordinate system
        #pivot_ref = SQ_MARKER_NAVIGATION_SUPPORT_LIB.apply_tranform_to_points(pivot_cam, T_ref_tib)
        #pivot_ref = pivot_ref[4, :]
        #pivot_ref = capture_median_pivot_point(duration=1.0, fps_estimate=30, use_kalman=True)
        pivot_ref = capture_average_pivot_point(T_ref_tib, duration=1, fps_estimate=25)
        #pivot_ref = corner_point[0][1]
        if pivot_ref is None:
            print("[Error] Failed to compute average pivot")
            return False

        #update_current_point("Capturing: Tibia Knee Center")
        # Capture point based on current count
        if len(tibia_list) == 0:  # KC (Knee Center)
            print("[Capture] KC - Knee Center")
            tibia_list.append(pivot_ref)
            relative_pos = get_relative_position(corner_point[2], pivot_ref)
            relative_tibia_points_list.append(relative_pos)
            
            print(f"Stored pivot_ref: {pivot_ref}")
            update_next_tibia_point()
            return True
            
        elif len(tibia_list) == 1:  # TMC (Tibia Medial Condyle)
            print("[Capture] TMC - Tibia Medial Condyle")
            tibia_list.append(pivot_ref)
            relative_pos = get_relative_position(corner_point[2], pivot_ref)
            relative_tibia_points_list.append(relative_pos)
            
            print(f"Stored pivot_ref: {pivot_ref}")
            update_next_tibia_point()
            return True
            
        elif len(tibia_list) == 2:  # TLC (Tibia Lateral Condyle)
            print("[Capture] TLC - Tibia Lateral Condyle")
            tibia_list.append(pivot_ref)
            relative_pos = get_relative_position(corner_point[2], pivot_ref)
            relative_tibia_points_list.append(relative_pos)
            
            print(f"Stored pivot_ref: {pivot_ref}")
            update_next_tibia_point()
            return True
            
        elif len(tibia_list) == 3:  # Tuberosity
            print("[Capture] Tuberosity")
            tibia_list.append(pivot_ref)
            relative_pos = get_relative_position(corner_point[2], pivot_ref)
            relative_tibia_points_list.append(relative_pos)
            print(f"Stored pivot_ref: {pivot_ref}")
            update_next_tibia_point()
            return True
            
        elif len(tibia_list) == 4:  # PCL
            print("[Capture] PCL")
            tibia_list.append(pivot_ref)
            relative_pos = get_relative_position(corner_point[2], pivot_ref)
            relative_tibia_points_list.append(relative_pos)
            print(f"Stored pivot_ref: {pivot_ref}")
            update_next_tibia_point()
            return True
            
        elif len(tibia_list) == 5:  # AC (Ankle Center) - needs 2 malleolus points
            if len(ac_Fetcher_List) == 0:
                print("[Capture] Medial Malleolus (AC Point 1/2)")
            else:
                print("[Capture] Lateral Malleolus (AC Point 2/2)")
            
            ac_Fetcher_List.append(pivot_ref)
            update_next_tibia_point()
            print(f"AC Fetcher List: {ac_Fetcher_List}")
            
            if len(ac_Fetcher_List) == 2:
                print("[Capture] AC - Ankle Center (calculating from 2 points)")
                
                Dist_medial_lateral = np.linalg.norm(ac_Fetcher_List[0] - ac_Fetcher_List[1])
                print(f"Distance medial-lateral: {Dist_medial_lateral}")
                
                ankle_center = findAnkleCenter(ac_Fetcher_List[0], ac_Fetcher_List[1])
                tibia_list.append(ankle_center)
                
                # Get relative position for ankle center
                relative_pos = get_relative_position(corner_point[2], ankle_center)
                relative_tibia_points_list.append(relative_pos)
                
                print("[Complete] All tibia points captured!")
                print(f"Final tibia_list: {tibia_list}")
                
                update_next_tibia_point()
                # Pass ac_Fetcher_List to XML function BEFORE clearing it
                save_tibia_distances_to_xml(tibia_list, ac_Fetcher_List, "save_xml/tibia_distances.xml")
                
                # Clear AC fetcher list AFTER saving
                ac_Fetcher_List = []
                return True
            return True
        
        else:
            print("[Warning] All tibia points already captured")
            
            return False
            
    except Exception as e:
        print(f"[Error] Computing pivot point: {e}")
        import traceback
        traceback.print_exc()
        return False


# def tibia_verifycuts():
#     global tibia_list,tibiaCount,Sig_Dir,ax,drawPlot,Sig_Dir,var,varAngle_combined,varflex_combined,Tmc_distance,Tlc_distance,CT_side
 
#     #global coronal_Dir
#     # Only proceed if tibia_list has at least 2 elements
#     if len(tibia_list) >= 6 and not (corner_point[1] == zeroArr).all():
#         # Points
#         knee_center = tibia_list[0]
#         tuberosity = tibia_list[3]
#         pcl = tibia_list[4]

#         # 1. Sagittal plane: normal is vector from tuberosity to PCL, passes through knee center
#         sagittal_normal = np.array(pcl) - np.array(tuberosity)
#         sagittal_normal = sagittal_normal / np.linalg.norm(sagittal_normal)
#         sagittal_plane = Planes_genrate_LIB.plane_from_point_and_normal(knee_center, sagittal_normal)

#         # 2. Axial plane: perpendicular to sagittal, passes through knee center
#         # Use mechanical axis (KC to AC) as reference direction
#         mechanical_axis = np.array(tibia_list[5]) - np.array(knee_center)
#         mechanical_axis = mechanical_axis / np.linalg.norm(mechanical_axis)
#         axial_normal = np.cross(sagittal_normal, mechanical_axis)
#         axial_normal = axial_normal / np.linalg.norm(axial_normal)
#         axial_plane = Planes_genrate_LIB.plane_from_point_and_normal(knee_center, axial_normal)

#         # 3. Coronal plane: perpendicular to both, passes through knee center
#         coronal_normal = np.cross(axial_normal, sagittal_normal)
#         coronal_normal = coronal_normal / np.linalg.norm(coronal_normal)
#         coronal_plane = Planes_genrate_LIB.plane_from_point_and_normal(knee_center, coronal_normal)

#         # Now you can use these planes for further calculations
#         # Example: angles between planes (should be ~90°)
#         angle_sag_ax = Planes_genrate_LIB.angle_between_planes(sagittal_plane, axial_plane)
#         angle_sag_cor = Planes_genrate_LIB.angle_between_planes(sagittal_plane, coronal_plane)
#         angle_ax_cor = Planes_genrate_LIB.angle_between_planes(axial_plane, coronal_plane)
#         #print(f"Angles (deg): Sag-Ax={angle_sag_ax:.2f}, Sag-Cor={angle_sag_cor:.2f}, Ax-Cor={angle_ax_cor:.2f}")

#         # varus_valgus
#         #TibiamechanicalAxis_Normal = Planes_genrate_LIB.project_on_plane(Dir,coronal_Dir)
#         TibiamechanicalAxis_Normal = Planes_genrate_LIB.project_on_plane(axial_normal,coronal_normal)
#         #TibiamechanicalAxis_Minus_Normal = Planes_genrate_LIB.project_on_plane(-Dir,coronal_Dir)
#         TibiamechanicalAxis_Minus_Normal = Planes_genrate_LIB.project_on_plane(-axial_normal,coronal_normal)
#         TibiaMarker_Normal = Planes_genrate_LIB.project_on_plane(vp_normal,coronal_normal)
#         TibiaMarker_Minus_Normal = Planes_genrate_LIB.project_on_plane(-vp_normal,coronal_normal)
#         varAngle = Planes_genrate_LIB.signed_angle((TibiamechanicalAxis_Normal-TibiamechanicalAxis_Minus_Normal),(TibiaMarker_Normal-TibiaMarker_Minus_Normal),coronal_normal)[0]
#         print("varangle:",varAngle)
#         # if varAngle < 0:
#         #     if CT_side == "L":
#         #         #print("Varus Angle: ",-varAngle)
#         #         cv2.putText(left,f"Coronal: {(-varAngle):.0f} valgus", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
#         #     else:
#         #         #print("Valgus Angle: ",-varAngle)
#         #         cv2.putText(left,f"Coronal: {(-varAngle):.0f} varus", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
#         # else:
#         #     if CT_side == "L":
#         #         #print("Valgus Angle: ",varAngle)
#         #         cv2.putText(left,f"Coronal: {(varAngle):.0f} varus", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
#         #     else:
#         #         #print("Varus Angle: ",varAngle)
#         #         cv2.putText(left,f"Coronal: {(varAngle):.0f} valgus", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
#         #Ante/Post
#         TibiamechanicalAxis_Normal = Planes_genrate_LIB.project_on_plane(axial_normal,sagittal_normal)
#         TibiamechanicalAxis_Minus_Normal = Planes_genrate_LIB.project_on_plane(-axial_normal,sagittal_normal)
#         TibiaMarker_Normal = Planes_genrate_LIB.project_on_plane(vp_normal,sagittal_normal) 
#         TibiaMarker_Minus_Normal = Planes_genrate_LIB.project_on_plane(-vp_normal,sagittal_normal)
#         varflex = Planes_genrate_LIB.signed_angle((TibiamechanicalAxis_Normal-TibiamechanicalAxis_Minus_Normal),(TibiaMarker_Normal-TibiaMarker_Minus_Normal),sagittal_normal)[0]
#         print("varflex:",varflex)
#         #varFlex = Planes_genrate_LIB.signed_angle((TibiamechanicalAxis_Normal-TibiamechanicalAxis_Minus_Normal),(TibiaMarker_Normal-TibiaMarker_Minus_Normal),Sig_Dir)[1] 
#         #print(" TIBiA Ant Angle signed: ",varflex)
#         #print(" TIBiA Flex Angle : ",varFlex)
#         # if varflex < 0:
#         #     #print("Ant Angle: ",-varflex)
#         #     if CT_side == "L":
#         #         cv2.putText(left,f"Sagital: {(-varflex):.0f} Ant", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
#         #     else:
#         #         cv2.putText(left,f"Sagital: {(-varflex):.0f} Post", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
#         # else:
#         #     if CT_side == "L":
#         #         cv2.putText(left,f"Sagital: {(varflex):.0f} Post", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
#         #     else:
#         #         cv2.putText(left,f"Sagital: {(varflex):.0f} Ant", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
#         Tmc_distance = -int(Planes_genrate_LIB.point_plane_distance(tibia_list[1],vp_normal,center_points[1]))
#         #print("TMC_distance before:",Tmc_distance)
#         #Tmc_distance = round(Tmc_distance, 0) 
#         #print("TMC_distance:",Tmc_distance)
#         Tlc_distance = -int(Planes_genrate_LIB.point_plane_distance(tibia_list[2],vp_normal,center_points[1]))
#         #Tlc_distance = round(Tlc_distance, 0)
#         #print("TLC_distance:",Tlc_distance)

#         if varAngle < 0:
#             varAngle_name = "Val" if CT_side == "L" else "Val"
#         else:
#             varAngle_name = "Var" if CT_side == "L" else "Var"

#         # if varflex < 0:
#         #     varflex_name = "Ant" if CT_side == "L" else "Ant"
#         # else:
#         #     varflex_name = "Post" if CT_side == "L" else "Post"

#         if CT_side == "L":
#             display_angle = -varflex  # Invert for left side
#         else:
#             display_angle = varflex

#         # Display angle with consistent terminology
#         if display_angle < 0:
#             # cv2.putText(left, f"Sagital: {abs(display_angle):.0f} Ant", 
#             #         (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
#             varflex_name = "Ant"
#         else:
#             # cv2.putText(left, f"Sagital: {abs(display_angle):.0f} Post", 
#             #         (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
#             varflex_name = "Post"
                
#         # Combine names and values into formatted strings
#         varAngle_combined = f" {abs(varAngle):.0f} {varAngle_name}"
#         varflex_combined = f" {abs(display_angle):.0f} {varflex_name}"

#         # # Encode the combined strings to fixed-length byte strings
#         # varAngle_encoded = varAngle_combined.encode('utf-8').ljust(32, b'\x00')  # 32 bytes
#         # varflex_encoded = varflex_combined.encode('utf-8').ljust(32, b'\x00')    # 32 bytes

#         # Pack the buffer with the new fields
#         # buffer = struct.pack(
#         #     '16s8s32s32sff',
#         #     selected_object_encoded,
#         #     ct_side_encoded,
#         #     varAngle_encoded,
#         #     varflex_encoded,
#         #     Tlc_distance,
#         #     Tmc_distance,
#         # )
        
#         # cv2.putText(left, f"medial: {Tmc_distance:.0f} mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
#         # cv2.putText(left, f"lateral: {Tlc_distance:.0f} mm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
#         #print(tibia_list)

def tibia_verifycuts():
    global sagittal_plane, axial_plane, coronal_plane, vp_normal, prev_vp_normal, prev_center
    global sagittal_normal, axial_normal, coronal_normal
    global tibia_list, tibiaCount, Sig_Dir, ax, drawPlot, Sig_Dir, var, varAngle_combined, varflex_combined
    global Tmc_distance, Tlc_distance, CT_side, sagittal_plane, coronal_plane, axial_plane
 
    # Only proceed if tibia_list has at least 6 elements
    if len(tibia_list) >= 6 and not (corner_point[1] == zeroArr).all():
        # Extract points from tibia_list
        kc_point = tibia_list[0]      # KC (Knee Center)
        tmc_point = tibia_list[1]     # TMC
        tlc_point = tibia_list[2]     # TLC
        tuberosity_point = tibia_list[3]  # Tuberosity
        pcl_point = tibia_list[4]     # PCL
        ankle_center = tibia_list[5]  # Ankle Center (AC)
        
        # Create the mechanical axis (from KC to Ankle Center)
        mechanical_axis_vector = ankle_center - kc_point
        mechanical_axis_vector = mechanical_axis_vector / np.linalg.norm(mechanical_axis_vector)
        
        # Generate sagittal plane using tuberosity and PCL points
        if tibiaCount is None:
            # Create vector from tuberosity to PCL
            tuberosity_to_pcl = pcl_point - tuberosity_point
            tuberosity_to_pcl = tuberosity_to_pcl / np.linalg.norm(tuberosity_to_pcl)
                
            # Create sagittal plane normal
            sagittal_normal = tuberosity_to_pcl / np.linalg.norm(tuberosity_to_pcl)
            sagittal_plane = Plane(tuple(kc_point), normal_vector=sagittal_normal)

            # Axial normal: mechanical axis (KC to AC)
            axial_normal = mechanical_axis_vector / np.linalg.norm(mechanical_axis_vector)
            axial_plane = Plane(tuple(kc_point), normal_vector=axial_normal)

            # Coronal normal: cross product of sagittal and axial normals
            coronal_normal = np.cross(sagittal_normal, axial_normal)
            coronal_normal = coronal_normal / np.linalg.norm(coronal_normal)
            coronal_plane = Plane(tuple(kc_point), normal_vector=coronal_normal)

            # Check angles
            angle_sagittal_coronal = Planes_genrate_LIB.angle_between_planes(sagittal_plane, coronal_plane)
            angle_sagittal_axial = Planes_genrate_LIB.angle_between_planes(sagittal_plane, axial_plane)
            angle_coronal_axial = Planes_genrate_LIB.angle_between_planes(coronal_plane, axial_plane)

            #print(f"Angle between sagittal and coronal: {angle_sagittal_coronal:.1f} degrees")
            #print(f"Angle between sagittal and axial: {angle_sagittal_axial:.1f} degrees")
            #print(f"Angle between coronal and axial: {angle_coronal_axial:.1f} degrees")
            tibiaCount = 1
        
        # Varus/Valgus calculation (in coronal plane)
        TibiamechanicalAxis_Normal = Planes_genrate_LIB.project_on_plane(mechanical_axis_vector, sagittal_normal)
        TibiamechanicalAxis_Minus_Normal = Planes_genrate_LIB.project_on_plane(-mechanical_axis_vector, sagittal_normal)
        TibiaMarker_Normal = Planes_genrate_LIB.project_on_plane(vp_normal, sagittal_normal)
        TibiaMarker_Minus_Normal = Planes_genrate_LIB.project_on_plane(-vp_normal, sagittal_normal)
        varAngle = Planes_genrate_LIB.signed_angle((TibiamechanicalAxis_Normal-TibiamechanicalAxis_Minus_Normal),(TibiaMarker_Normal-TibiaMarker_Minus_Normal), sagittal_normal)[0]
        #print("varAngle", varAngle)
        
        # Anterior/Posterior calculation (in sagittal plane)
        TibiamechanicalAxis_Normal = Planes_genrate_LIB.project_on_plane(mechanical_axis_vector, coronal_normal)
        TibiamechanicalAxis_Minus_Normal = Planes_genrate_LIB.project_on_plane(-mechanical_axis_vector, coronal_normal)
        TibiaMarker_Normal = Planes_genrate_LIB.project_on_plane(vp_normal, coronal_normal) 
        TibiaMarker_Minus_Normal = Planes_genrate_LIB.project_on_plane(-vp_normal, coronal_normal)
        varflex = Planes_genrate_LIB.signed_angle((TibiamechanicalAxis_Normal-TibiamechanicalAxis_Minus_Normal),(TibiaMarker_Normal-TibiaMarker_Minus_Normal), coronal_normal)[0]
        #print("varflex", varflex)
        
        # Distance calculations
        vp_normal = smooth_normal(prev_vp_normal, vp_normal)
        plane_point = smooth_point(prev_center, center_points[1])
        #print("tmc_point:", tmc_point)
        #print("tlc_point:", tlc_point)
        #print("vp_normal:", vp_normal)
        #print("center_points[1]:", center_points[1])
        Tmc_distance = -int(Planes_genrate_LIB.point_plane_distance(tmc_point, vp_normal, plane_point))
        Tlc_distance = -int(Planes_genrate_LIB.point_plane_distance(tlc_point, vp_normal, plane_point))
        #print("TMC distance:", Tmc_distance)
        #print("TLC distance:", Tlc_distance)
        
        prev_vp_normal, prev_center = vp_normal, plane_point
        
        # Determine naming convention based on CT side and angle values
        if varAngle < 0:
            varAngle_name = "Varus" if CT_side == "L" else "Valgus"
        else:
            varAngle_name = "Valgus" if CT_side == "L" else "Varus"
        
        # Display angle with consistent terminology
        if varflex < 0:
            varflex_name = "Anti" if CT_side == "L" else "Anti"
        else:
            varflex_name = "Post" if CT_side == "L" else "Post"
                
        # Combine names and values into formatted strings
        varAngle_combined = f" {abs(varAngle):.0f} {varAngle_name}"
        varflex_combined = f" {abs(varflex):.0f} {varflex_name}"
        
        # Display on frame (for local OpenCV window)
        cv2.putText(left, f"Var/Valgus: {varAngle_combined} ", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(left, f"Slope: {varflex_combined} ", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(left, f"medial: {Tmc_distance +7:.0f} mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(left, f"lateral: {Tlc_distance +7:.0f} mm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # ==================== SEND RESULTS TO WEB ====================
        try:
            # Import the function (already imported at top of main file)
            update_results(
                var_angle=varAngle,
                var_flex=varflex,
                tmc_distance=Tmc_distance+7,
                tlc_distance=Tlc_distance+7,
                var_angle_name=varAngle_name,
                var_flex_name=varflex_name
            )
            print(f"[Web] Results sent: {varAngle_combined}, {varflex_combined}")
        except Exception as e:
            print(f"[Web] Could not send results: {e}")
        # ==================== END WEB UPDATE ====================

def femur_point_referencing():
    global relative_femur_points_list
    for index,tp in enumerate(relative_femur_points_list):
        # Get new child position
        new_child_position = update_child_position(corner_point[3], tp)
        femur_list[index] = new_child_position

def tibia_point_referencing():
    global relative_tibia_points_list
    for index,tp in enumerate(relative_tibia_points_list):
        # Get new child position
        new_child_position = update_child_position(corner_point[2], tp)
        tibia_list[index] = new_child_position
        
def draw3dPoseMarkers(camera_matrix, dist_coeffs, img, corners, ids, marker_length):
    # Get rotation and translation vectors
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

    # Draw axis for visualization
    for i in range(len(ids)):
        cv2.aruco.drawAxis(img, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.05)


def femur_pointtaker_o():
    global femur_list
    global hip_center_fetcher_list
    global is_taking_Hip_Center
    global removedFirstValue
    global prev_button_a_femur
    global last_button_a_trigger_femur

    pointer_corners = corner_point[0]    
    try:
        
        # Compute pivot point in camera coordinates
        pivot_cam = pivot_point_obj.get_pivot_point_CamCoordSys(pointer_corners)
        pivot_ref = SQ_MARKER_NAVIGATION_SUPPORT_LIB.apply_tranform_to_points(pivot_cam, T_ref_fem) #taking pointer corners for now,
        pivot_ref = pivot_ref[4,:]
        
        
        if is_taking_Hip_Center and femurFirstFrame is not None:
            # Transform to reference coordinate system
            femur_refernce = getRelativeToMarker(corner_point[3],corner_point[1])
            refernce_femur = getRelativeToMarker(femur_refernce,femurFirstFrame)
            
        # Check input conditions
        key = cv2.waitKey(1) & 0xFF
        #pygame.event.pump()  # Poll joystick events
        #current_time = pygame.time.get_ticks()

        # Get current button state
        #button_a_current = joystick.get_button(0)

        # Detect rising edge (transition from not pressed to pressed)
        #button_a_pressed = button_a_current and not prev_button_a_femur
        #prev_button_a_femur = button_a_current  # Update previous state

        # Check trigger conditions
        #key_pressed = (key == ord('c'))
        #button_allowed = button_a_pressed and (current_time - last_button_a_trigger_femur >= 1000)
        
        if key == ord('c') and femurFirstFrame is not None and len(femur_list) <= 6:
            # Update cooldown if triggered by button
            #if button_a_pressed:
                #last_button_a_trigger_femur = current_time
                
            print("Pressed C")
            if len(femur_list) == 0:
                print("HC")
                # if not (corner_point[0] == zeroArr).all() and not (corner_point[3] == zeroArr).all():
                #     print("HC")
                #     femur_list.append(pivot_ref)
                #     relative_pos = get_relative_position(corner_point[3], pivot_ref)
                #     relative_femur_points_list.append(relative_pos)
                #     print(f"Stored pivot_ref: {pivot_ref}")
                #     print(f"Current list: {femur_list}")
                ##HC Finder with 5 points.
                if not (corner_point[3] == zeroArr).all():
                #if not (corner_point[3] == zeroArr).all():
                    hip_center_fetcher_print_list.append(np.asarray(refernce_femur))
                    hip_center_fetcher_list.append(np.asarray(refernce_femur.T))
                    print(hip_center_fetcher_list)
                    if(len(hip_center_fetcher_list) == 5):
                        print(hip_center_fetcher_list)
                        #np.save("Hip_center_corner_points",hip_center_fetcher_list)
                        #Find HipCenter
                        pp = np.asarray(hip_center_fetcher_list)
                        print("\n\n")
                        print(hip_center_fetcher_print_list)
                        print("\n\n")
                        print("\n\n")
                        print(pp)
                        print("\n\n")

                        hipCenter = get_pivot_point(pp)["pv_point"]
                        print("Calculated Hip Center:", hipCenter)  # Check computed center
                        #draw_a_plot_CornerList(hip_center_fetcher_print_list, corner_point[1], hipCenter)
                        femur_list.append(hipCenter)
                        relative_pos = get_relative_position(corner_point[3], hipCenter)
                        print("realtive pos:",relative_pos)
                        relative_femur_points_list.append(relative_pos)
                        hip_center_fetcher_list = []
                        print("HC finder Points taken")
                        print("HC Found")
                        #print(femur_list)
                    
                        is_taking_Hip_Center = False
            elif len(femur_list) == 1:
                T_ref_vf = None
                if not (corner_point[0] == zeroArr).all() and not (corner_point[3] == zeroArr).all():
                    print("FKC")
                    femur_list.append(pivot_ref)
                    relative_pos = get_relative_position(corner_point[3], pivot_ref)
                    relative_femur_points_list.append(relative_pos)
                    print(f"Stored pivot_ref: {pivot_ref}")
                    print(f"Current list: {femur_list}")
            elif len(femur_list) == 2:
                if not (corner_point[0] == zeroArr).all() and not (corner_point[3] == zeroArr).all():
                    print("DMP")
                    femur_list.append(pivot_ref)
                    relative_pos = get_relative_position(corner_point[3], pivot_ref)
                    relative_femur_points_list.append(relative_pos)
                    print(f"Stored pivot_ref: {pivot_ref}")
                    print(f"Current list: {femur_list}")
            elif len(femur_list) == 3:
                if not (corner_point[0] == zeroArr).all() and not (corner_point[3] == zeroArr).all():
                    print("DLP")
                    femur_list.append(pivot_ref)
                    relative_pos = get_relative_position(corner_point[3], pivot_ref)
                    relative_femur_points_list.append(relative_pos)
                    print(f"Stored pivot_ref: {pivot_ref}")
                    print(f"Current list: {femur_list}")
            elif len(femur_list) == 4:
                if not (corner_point[0] == zeroArr).all() and not (corner_point[3] == zeroArr).all():
                    print("Surgical_Medial")
                    femur_list.append(pivot_ref)
                    relative_pos = get_relative_position(corner_point[3], pivot_ref)
                    relative_femur_points_list.append(relative_pos)
                    print(f"Stored pivot_ref: {pivot_ref}")
                    print(f"Current list: {femur_list}")
            elif len(femur_list) == 5:
                if not (corner_point[0] == zeroArr).all() and not (corner_point[3] == zeroArr).all():
                    print("Surgical_lateral")
                    femur_list.append(pivot_ref)
                    relative_pos = get_relative_position(corner_point[3], pivot_ref)
                    relative_femur_points_list.append(relative_pos)
                    print(f"Stored pivot_ref: {pivot_ref}")
                    print(f"Current list: {femur_list}")

    except Exception as e:
        print(f"Error computing pivot point: {e.with_traceback()}")

def femur_verifycuts():
    global femur_list,femurCount,Sig_Dir_femur,ax,drawPlot,coronal_Dir,sagittal_plane,Dir,Dmp_distance,Dlp_distance,varAngle_combined,varflex_combined
    global sagittal_plane, axial_plane, coronal_plane,prev_vp_normal, prev_center,vp_normal
    global sagittal_normal_femur, axial_normal_femur, coronal_normal_femur
    # Only proceed if tibia_list has at least 2 elements
    if len(femur_list) >= 6 and not (corner_point[1] == zeroArr).all():
        hip_center = femur_list[0]      # Hip Center
        knee_center = femur_list[1]     # Knee Center
        dmp = femur_list[2]             # Distal Medial Condyle
        dlp = femur_list[3]             # Distal Lateral Condyle
        surgical_medial = femur_list[4] # Surgical Medial Condyle
        surgical_lateral = femur_list[5]# Surgical Lateral Condyle
        if femurCount is None:
            # 1. Coronal plane: normal is vector from surgical medial to surgical lateral, passes through knee center
            coronal_normal_femur = np.array(surgical_lateral) - np.array(surgical_medial)
            coronal_normal_femur = coronal_normal_femur / np.linalg.norm(coronal_normal_femur)
            coronal_plane = Planes_genrate_LIB.plane_from_point_and_normal(knee_center, coronal_normal_femur)

            # 2. Axial plane: normal is vector from hip center to knee center, passes through knee center
            axial_normal_femur = np.array(hip_center) - np.array(knee_center)
            axial_normal_femur = axial_normal_femur / np.linalg.norm(axial_normal_femur)
            axial_plane = Planes_genrate_LIB.plane_from_point_and_normal(knee_center, axial_normal_femur)

            # 3. Sagittal plane: cross product of coronal and axial normals, passes through knee center
            sagittal_normal_femur = np.cross(coronal_normal_femur, axial_normal_femur)
            sagittal_normal_femur = sagittal_normal_femur / np.linalg.norm(sagittal_normal_femur)
            sagittal_plane = Planes_genrate_LIB.plane_from_point_and_normal(knee_center, sagittal_normal_femur)

            # Optionally print angles for debug
            angle_coronal_axial = Planes_genrate_LIB.angle_between_planes(coronal_plane, axial_plane)
            angle_coronal_sagittal = Planes_genrate_LIB.angle_between_planes(coronal_plane, sagittal_plane)
            angle_axial_sagittal = Planes_genrate_LIB.angle_between_planes(axial_plane, sagittal_plane)
            print(f"Angle between coronal and axial: {angle_coronal_axial:.1f} degrees")
            print(f"Angle between coronal and sagittal: {angle_coronal_sagittal:.1f} degrees")
            print(f"Angle between axial and sagittal: {angle_axial_sagittal:.1f} degrees")
            femurCount = 1
     

        # varus_valgus
        FemurmechanicalAxis_Normal = Planes_genrate_LIB.project_on_plane(axial_normal_femur,sagittal_normal_femur)
        FemurmechanicalAxis_Minus_Normal = Planes_genrate_LIB.project_on_plane(-axial_normal_femur,sagittal_normal_femur)
        FemurMarker_Normal = Planes_genrate_LIB.project_on_plane(vp_normal,sagittal_normal_femur)
        FemurMarker_Minus_Normal = Planes_genrate_LIB.project_on_plane(-vp_normal,sagittal_normal_femur)
        varAngle = Planes_genrate_LIB.signed_angle((FemurmechanicalAxis_Normal-FemurmechanicalAxis_Minus_Normal),(FemurMarker_Normal-FemurMarker_Minus_Normal),sagittal_normal_femur)[0]
       
        
        #Ante/Post
        FemurmechanicalAxis_Normal = Planes_genrate_LIB.project_on_plane(axial_normal_femur,coronal_normal_femur)
        FemurmechanicalAxis_Minus_Normal = Planes_genrate_LIB.project_on_plane(-axial_normal_femur,coronal_normal_femur)
        FemurMarker_Normal = Planes_genrate_LIB.project_on_plane(vp_normal,coronal_normal_femur)
        FemurMarker_Minus_Normal = Planes_genrate_LIB.project_on_plane(-vp_normal,coronal_normal_femur)
        varflex = Planes_genrate_LIB.signed_angle((FemurmechanicalAxis_Normal-FemurmechanicalAxis_Minus_Normal),(FemurMarker_Normal-FemurMarker_Minus_Normal),coronal_normal_femur)[0]
        print("femur varflex angle: ",varflex)
        
        vp_normal = smooth_normal(prev_vp_normal, vp_normal)
        plane_point = smooth_point(prev_center, center_points[1])
        
        Dmp_distance = -int(Planes_genrate_LIB.point_plane_distance(femur_list[2],vp_normal,plane_point))
        Dlp_distance = -int(Planes_genrate_LIB.point_plane_distance(femur_list[3],vp_normal,plane_point))
        
        prev_vp_normal, prev_center = vp_normal, plane_point
        
        if varAngle < 0:
            varAngle_name = "Varus" if CT_side == "L" else "Varus"
        else:
            varAngle_name = "Valgus" if CT_side == "L" else "Valgus"

        # if varflex < 0:
        #     varflex_name = "ext" if CT_side == "L" else "ext"
        # else:
        #     varflex_name = "flex" if CT_side == "L" else "flex"
        """
        if CT_side == "L":
            display_angle = -varflex  # Invert for left side
        else:
            display_angle = varflex

        if display_angle < 0:
            # cv2.putText(left, f"Sagital: {abs(display_angle):.0f} Ant", 
            #         (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            varflex_name = "flex"
        else:
            # cv2.putText(left, f"Sagital: {abs(display_angle):.0f} Post", 
            #         (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            varflex_name = "ext"
        """   
        if CT_side == "L":
            # Left side: negative = flexion, positive = extension
            display_angle = varflex
            if varflex < 0:
                varflex_name = "ext"
            else:
                varflex_name = "flex"
        else:  # CT_side == "R"
            # Right side: INVERTED - negative = extension, positive = flexion
            display_angle = -varflex  # Keep original sign
            if varflex < 0:
                varflex_name = "flex"   # Inverted for right side
            else:
                varflex_name = "ext"  # Inverted for right side

        # Combine names and values into formatted strings
        varAngle_combined = f"{abs(varAngle):.0f} {varAngle_name}"
        varflex_combined = f" {abs(varflex):.0f} {varflex_name}"
        
        cv2.putText(left, f"Var/Valgus: {varAngle_combined} ", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(left, f"Slope: {varflex_combined} ", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        
        cv2.putText(left, f"medial: {Dmp_distance + 7:.0f} mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(left, f"lateral: {Dlp_distance + 7:.0f} mm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # ==================== SEND RESULTS TO WEB ====================
        try:
            # Import the function (already imported at top of main file)
            update_results(
                var_angle=varAngle,
                var_flex=varflex,
                tmc_distance=Dmp_distance+7,
                tlc_distance=Dlp_distance+7,
                var_angle_name=varAngle_name,
                var_flex_name=varflex_name
            )
            print(f"[Web] Results sent: {varAngle_combined}, {varflex_combined}")
        except Exception as e:
            print(f"[Web] Could not send results: {e}")
        # ==================== END WEB UPDATE ====================


def femur_pointtaker_main():
    global femur_list
    global hip_center_fetcher_list
    global hip_center_fetcher_print_list
    global is_taking_Hip_Center
    global removedFirstValue
    global prev_button_a_femur
    global last_button_a_trigger_femur
    global relative_femur_points_list
    global femurFirstFrame  # ← This is the global variable name

    try:
        # ===== CHECK FIRST FRAME =====
        # CRITICAL: femurFirstFrame must be set before collecting hip center
        if femurFirstFrame is None and is_taking_Hip_Center:
            print("[Femur] ❌ femurFirstFrame not set. Press 'w' to capture first frame!")
            print(f"[Debug] femurFirstFrame = {femurFirstFrame}")
            print(f"[Debug] is_taking_Hip_Center = {is_taking_Hip_Center}")
            return False

        pointer_corners = corner_point[0]
        
        # Check if pointer marker detected
        if (corner_point[0] == zeroArr).all():
            print("[Femur] Pointer marker not detected")
            return False
            
        pivot_ref = capture_average_pivot_point(duration=1.0, fps_estimate=30)
        if pivot_ref is None:
            print("[Error] Failed to compute average pivot")
            return False    
        # Compute pivot point in camera coordinates
        #pivot_cam = pivot_point_obj.get_pivot_point_CamCoordSys(pointer_corners)
        #pivot_ref = SQ_MARKER_NAVIGATION_SUPPORT_LIB.apply_tranform_to_points(pivot_cam, T_ref_fem)
        #pivot_ref = pivot_ref[4, :]

        # If we're in hip-center-collection mode, compute the reference frame value
        if is_taking_Hip_Center and femurFirstFrame is not None:
            # Check if verification marker (ID 5) is detected
            if (corner_point[1] == zeroArr).all():
                print("[Femur] Verification marker not detected for hip center collection")
                return False
                
            femur_refernce = getRelativeToMarker(corner_point[3], corner_point[1])
            refernce_femur = getRelativeToMarker(femur_refernce, femurFirstFrame)
            print(f"[Debug] Reference calculated for hip center sample")

        # If already have 6 femur points, don't add more
        if len(femur_list) >= 6:
            print("[Femur] ✓ Already collected 6 femur points")
            return False

        # ========== HIP CENTER COLLECTION ==========
        if len(femur_list) == 0:
            # Check if verification and femur markers are detected
            if not (corner_point[3] == zeroArr).all() and not (corner_point[1] == zeroArr).all():
                # Append the sample
                hip_center_fetcher_print_list.append(np.asarray(refernce_femur))
                hip_center_fetcher_list.append(np.asarray(refernce_femur.T))
                print(f"[Femur] ✓ Hip center sample {len(hip_center_fetcher_list)}/5 captured")
                
                # When we have 5 samples, compute hip center
                if len(hip_center_fetcher_list) == 5:
                    pp = np.asarray(hip_center_fetcher_list)
                    print("[Femur] Computing hip center from 5 samples...")
                    print("[Femur] Hip-center samples (print list):")
                    print(hip_center_fetcher_print_list)
                    print("[Femur] Hip-center samples (array):")
                    print(pp)
                    
                    hipCenter = get_pivot_point(pp)["pv_point"]
                    print(f"[Femur] ✓ Hip Center calculated: {hipCenter}")
                    
                    femur_list.append(hipCenter)
                    relative_pos = get_relative_position(corner_point[3], hipCenter)
                    relative_femur_points_list.append(relative_pos)
                    
                    # Clear hip center buffers
                    hip_center_fetcher_list = []
                    hip_center_fetcher_print_list = []
                    is_taking_Hip_Center = False
                    
                    print("[Femur] ✓ Hip center finalized and stored as first femur point")
                    
                    # Update web
                    if WEB_MODE_ENABLED:
                        try:
                            update_point_count('femur', len(femur_list))
                        except Exception as e:
                            print(f"[Web] update_point_count error: {e}")
                    return True
                else:
                    # Still collecting hip-center samples
                    if WEB_MODE_ENABLED:
                        try:
                            update_point_count('femur', 0)  # Keep at 0 until hip center finalized
                        except Exception as e:
                            print(f"[Web] update_point_count error: {e}")
                    return True
            else:
                print("[Femur] ❌ Femur or verification marker not detected")
                return False
                
        # ========== REGULAR FEMUR POINTS (after hip center) ==========
        elif len(femur_list) >= 1 and len(femur_list) < 6:
            # Ensure markers present
            if not (corner_point[0] == zeroArr).all() and not (corner_point[3] == zeroArr).all():
                label_map = {
                    1: "Femur Knee Center",
                    2: "Distal Medial Condyle",
                    3: "Distal Lateral Condyle",
                    4: "Surgical Medial",
                    5: "Surgical Lateral"
                }
                label = label_map.get(len(femur_list), f"Femur point {len(femur_list)}")
                
                femur_list.append(pivot_ref)
                relative_pos = get_relative_position(corner_point[3], pivot_ref)
                relative_femur_points_list.append(relative_pos)
                
                print(f"[Femur] ✓ {label} captured: {pivot_ref}")
                print(f"[Femur] Total points: {len(femur_list)}/6")

                if WEB_MODE_ENABLED:
                    try:
                        update_point_count('femur', len(femur_list))
                    except Exception as e:
                        print(f"[Web] update_point_count error: {e}")

                return True
            else:
                print("[Femur] ❌ Pointer or femur marker not detected")
                return False

        # Shouldn't reach here
        print("[Femur] No action taken")
        return False

    except Exception as e:
        print(f"[Femur] ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def update_next_femur_point():
    """Update the UI with the next femur point to capture."""
    global femurFirstFrame, is_taking_Hip_Center, hip_center_fetcher_list, femur_list
    
    try:
        if femurFirstFrame is None:
            update_current_point("Ready for Femur First Frame")
        elif is_taking_Hip_Center:
            count = len(hip_center_fetcher_list)
            if count < 5:
                update_current_point(f"Hip Center {count + 1}/5")
            else:
                # This state is brief, as it finalizes automatically
                update_current_point("Finalizing Hip Center...")
        else:
            # Hip center is done, now anatomical points
            count = len(femur_list) # count will be 1 after hip center
            if count == 1:
                update_current_point("Femur Knee Center")
            elif count == 2:
                update_current_point("Distal Medial Condyle")
            elif count == 3:
                update_current_point("Distal Lateral Condyle")
            elif count == 4:
                update_current_point("Surgical Medial")
            elif count == 5:
                update_current_point("Surgical Lateral")
            elif count >= 6:
                update_current_point("Femur Points Complete")
    except Exception as e:
        print(f"[Femur] Error updating next point: {e}")


def femur_pointtaker():
    """
    Captures femur points sequentially, handling 'First Frame', 
    'Hip Center' (5 points), and 'Anatomical Points' (5 points)
    all from a single trigger.
    """
    global femur_list, hip_center_fetcher_list, hip_center_fetcher_print_list
    global is_taking_Hip_Center, relative_femur_points_list, femurFirstFrame
    global corner_point, zeroArr, pivot_point_obj, T_ref_fem, WEB_MODE_ENABLED
    
    try:
        # ===== 1. CAPTURE FIRST FRAME (if not already set) =====
        # This is the FIRST action when femur is selected
        # REQUIRES: Femur (13) and Verification (5)
        if femurFirstFrame is None and is_taking_Hip_Center:
            # Check if verification (1) and femur (3) markers are visible
            if (corner_point[1] == zeroArr).all() or (corner_point[3] == zeroArr).all():
                print("[Femur] ? Verification (5) or Femur (13) marker not visible to capture first frame.")
                if WEB_MODE_ENABLED: update_current_point("Markers not visible for First Frame")
                return False
                
            # Capture the first frame
            femurFirstFrame = getRelativeToMarker(corner_point[3], corner_point[1])
            print(f"[Femur] ? First Frame Captured.")
            
            if WEB_MODE_ENABLED:
                update_next_femur_point() # Update UI to "Hip Center 1/5"
            return True # Indicate success
            
        # ===== 2. HIP CENTER COLLECTION (after first frame is set) =====
        # This is the SECOND action (steps 2-6). 
        # REQUIRES: Femur (13) and Verification (5)
        # *** NO POINTER NEEDED HERE ***
        if len(femur_list) == 0 and is_taking_Hip_Center:
            # Check if verification (1) and femur (3) markers are detected
            if (corner_point[3] == zeroArr).all() or (corner_point[1] == zeroArr).all():
                print("[Femur] ? Femur (13) or Verification (5) marker not detected for hip center collection")
                #if WEB_MODE_ENABLED: update_current_point("Markers not visible for Hip Center")
                return False
                
            # Calculate the reference frame value for this sample
            femur_refernce = getRelativeToMarker(corner_point[3], corner_point[1])
            refernce_femur = getRelativeToMarker(femur_refernce, femurFirstFrame)
            
            # Append the sample
            hip_center_fetcher_print_list.append(np.asarray(refernce_femur))
            hip_center_fetcher_list.append(np.asarray(refernce_femur.T))
            print(f"[Femur] ? Hip center sample {len(hip_center_fetcher_list)}/5 captured")
            
            if WEB_MODE_ENABLED:
                update_next_femur_point() # Update UI to "Hip Center 2/5", etc.

            # When we have 5 samples, compute hip center
            if len(hip_center_fetcher_list) == 5:
                pp = np.asarray(hip_center_fetcher_list)
                print("[Femur] Computing hip center from 5 samples...")
                
                hipCenter = get_pivot_point(pp)["pv_point"]
                print(f"[Femur] ? Hip Center calculated: {hipCenter}")
                
                femur_list.append(hipCenter)
                
                # Get relative position of the calculated hip center
                if (corner_point[3] == zeroArr).all():
                     print("[Warning] Femur marker not visible when saving Hip Center relative pos. Using last known transform.")
                
                relative_pos = get_relative_position(corner_point[3], hipCenter)
                relative_femur_points_list.append(relative_pos)
                
                # Clear hip center buffers and advance state
                hip_center_fetcher_list = []
                hip_center_fetcher_print_list = []
                is_taking_Hip_Center = False
                
                print("[Femur] ? Hip center finalized and stored as first femur point")
                
                if WEB_MODE_ENABLED:
                    update_point_count('femur', len(femur_list))
                    update_next_femur_point() # Update UI to "Ready for Femur Knee Center"
            return True
        
        # ===== 3. REGULAR FEMUR POINTS (after hip center) =====
        # This is the THIRD action (steps 7-11).
        # REQUIRES: Pointer (8) and Femur (13)
        # *** POINTER IS NOW REQUIRED ***
        elif len(femur_list) >= 1 and len(femur_list) < 6:
            # Check if POINTER (0) and femur (3) markers are detected
            if (corner_point[0] == zeroArr).all() or (corner_point[3] == zeroArr).all():
                print("[Femur] ? Pointer (8) or Femur (13) marker not detected")
                #if WEB_MODE_ENABLED: update_current_point("Markers not visible for point")
                return False
            
            # Get the averaged pivot point from the pointer
            # Pass T_ref_fem to the modified function
            pivot_ref = capture_average_pivot_point(T_ref_fem, duration=1.0, fps_estimate=30) # MODIFIED
            if pivot_ref is None:
                print("[Error] Failed to compute average pivot for Femur")
                return False

            label_map = {
                1: "Femur Knee Center",
                2: "Distal Medial Condyle",
                3: "Distal Lateral Condyle",
                4: "Surgical Medial",
                5: "Surgical Lateral"
            }
            label = label_map.get(len(femur_list), f"Femur point {len(femur_list)}")
            
            femur_list.append(pivot_ref)
            relative_pos = get_relative_position(corner_point[3], pivot_ref)
            relative_femur_points_list.append(relative_pos)
            
            print(f"[Femur] ? {label} captured: {pivot_ref}")
            print(f"[Femur] Total points: {len(femur_list)}/6")

            if WEB_MODE_ENABLED:
                try:
                    update_point_count('femur', len(femur_list))
                    update_next_femur_point() # Update UI to next point
                    save_femur_distances_to_xml(femur_list, "save_xml/femur_distances.xml")
                except Exception as e:
                    print(f"[Web] update_point_count error: {e}")

            return True
        
        else:
            # All points (6) are already captured
            print("[Femur] ? Already collected 6 femur points")
            if WEB_MODE_ENABLED:
                update_next_femur_point() # Update UI to "Complete"
                
            return False

    except Exception as e:
        print(f"[Femur] ? Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_relative_position(parent_matrix, child_point):
    """
    Computes the relative position of the child point with respect to the parent.
    
    Args:
        parent_matrix (numpy.ndarray): 4x3 matrix of the parent marker (converted inside).
        child_point (numpy.ndarray): 1x3 matrix representing the child’s position.

    Returns:
        numpy.ndarray: 1x3 relative position of the child.
    """
    parent_matrix = parent_matrix.T  # Convert from 4x3 to 3x4
    child_point = child_point.reshape(3, 1)  # Ensure correct shape (3x1)
    
    # Compute transformation matrix from parent
    T_mat = rigid_body_transform_info.compute_rigid_body_transform(parent_matrix, parent_matrix)  # Identity initially
    
    # Compute relative position
    relative_position = np.linalg.inv(T_mat) @ np.vstack((child_point, [[1]]))  # Homogeneous coordinates
    return relative_position[:3].flatten()  # Return as 1x3

def update_child_position(parent_matrix, relative_position):
    """
    Updates the child’s position based on the parent’s new transformation.
    
    Args:
        parent_matrix (numpy.ndarray): Updated 4x3 matrix of the parent marker.
        relative_position (numpy.ndarray): Stored relative position (1x3).

    Returns:
        numpy.ndarray: Updated absolute position of the child.
    """
    parent_matrix = parent_matrix.T  # Convert from 4x3 to 3x4
    
    # Compute new transformation matrix of the parent
    T_mat = rigid_body_transform_info.compute_rigid_body_transform(parent_matrix, parent_matrix)
    
    # Apply transformation to get new absolute position
    updated_position = T_mat @ np.vstack((relative_position.reshape(3, 1), [[1]]))
    return updated_position[:3].flatten()  # Return as 1x3

def compute_rigid_body_transform(source_points, target_points):
    """
    Computes 4x4 rigid transformation matrix between two point sets
    source_points: 3xN array of source points
    target_points: 3xN array of target points
    Returns 4x4 transformation matrix
    """
    # Center points
    src_centroid = np.mean(source_points, axis=1, keepdims=True)
    tgt_centroid = np.mean(target_points, axis=1, keepdims=True)

    # SVD-based rotation calculation
    H = (target_points - tgt_centroid) @ (source_points - src_centroid).T
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.T @ U.T

    # Translation calculation
    t = tgt_centroid - R @ src_centroid

    # Build 4x4 matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    
    return T

def calculate_angle(A, B):
    """Calculate the angle between two 3D vectors in degrees."""
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    cos_theta = dot_product / (norm_A * norm_B)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid precision errors
    return np.degrees(theta)

def calculate_signed_angle(A, B, normal_vector):
    """Calculate signed angle between two 3D vectors in degrees."""
    dot = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    cos_theta = dot / (norm_A * norm_B)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    
    # Calculate cross product for direction
    cross = np.cross(A, B)
    direction = np.sign(np.dot(normal_vector, cross))
    
    return np.degrees(theta) * direction

def calculate_distance(marker1, marker2):
    """
    Calculate the Euclidean distance between two markers in 3D space.

    Args:
        marker1 (numpy.ndarray): 3D coordinates of the first marker (e.g., [x1, y1, z1]).
        marker2 (numpy.ndarray): 3D coordinates of the second marker (e.g., [x2, y2, z2]).

    Returns:
        float: The Euclidean distance between the two markers.
    """
    marker1 = np.array(marker1)
    marker2 = np.array(marker2)
    distance = np.linalg.norm(marker1 - marker2)
    return distance

def compute_rigid_body_Transform(source, target):
    """
    Computes the rigid transformation between two sets of corresponding points.
    Returns 4x4 affine transformation matrix.
    """
    assert source.shape == target.shape, "Source and target must have the same shape"
    
    # Center the points
    centroid_source = np.mean(source, axis=1, keepdims=True)
    centroid_target = np.mean(target, axis=1, keepdims=True)
    src_centered = source - centroid_source
    trg_centered = target - centroid_target
    
    # SVD decomposition
    H = src_centered @ trg_centered.T
    U, _, Vt = np.linalg.svd(H)
    
    # Rotation matrix
    R = Vt.T @ U.T
    
    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Translation vector
    t = centroid_target - R @ centroid_source
    
    # Create affine matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, [3]] = t
    
    return T

def apply_affine_Transform(points, T):
    """Applies 4x4 affine transformation to 3xN points"""
    homogeneous = np.vstack([points.T, np.ones((1, points.shape[0]))])
    transformed = (T @ homogeneous)[:3, :]
    return transformed.T

def virtual_to_camera(point_virtual, camera_marker_corners):
    """
    Converts a point from virtual coordinate system to camera coordinates.
    
    Args:
        point_virtual (np.ndarray): 1x3 point in virtual coordinate system
        camera_marker_corners (np.ndarray): 4x3 detected marker corners in camera coordinates
    
    Returns:
        np.ndarray: 1x3 point in camera coordinates
    """
    # Define virtual marker in its local coordinate system
    virtual_marker_local = np.array([
        [-30.0, 30.0, 0],   # Top-left
        [30.0, 30.0, 0],    # Top-right
        [30.0, -30.0, 0],   # Bottom-right
        [-30.0, -30.0, 0]   # Bottom-left
    ])
    
    # Compute transformation matrix (3xN format)
    T = compute_rigid_body_Transform(
        source=virtual_marker_local.T,  # Convert to 3x4
        target=camera_marker_corners.T  # Convert to 3x4
    )
    
    # Apply transformation to the virtual point
    point_camera = apply_affine_Transform(
        points=point_virtual[np.newaxis, :],  # Convert to 1x3 array
        T=T
    )
    
    return point_camera.flatten()

#==============START single point relative to marker===================
def getRelativeToPoint(point_matrix, ref_marker):   
    T_mat = compute_rigid_body_TANSFORM(ref_marker, corner_point[3])
    local_point = apply_affine_TANSFORM_single_point(point_matrix, T_mat)
    return local_point

def compute_rigid_body_TANSFORM(ref_marker, known_marker):
    """
    Compute the 4x4 rigid-body transformation that maps ref_marker points to known_marker points.
    This uses the Kabsch algorithm: we compute centroids of each point set, center the points,
    and then use SVD on the cross-covariance to find the optimal rotation and translation&#8203;:contentReference[oaicite:0]{index=0}.
    The returned matrix is a homogeneous transform combining the rotation R and translation t&#8203;:contentReference[oaicite:1]{index=1}.

    Parameters:
        ref_marker (array-like, Nx3): Coordinates of the reference marker points.
        known_marker (array-like, Nx3): Coordinates of the corresponding points in the known frame.

    Returns:
        np.ndarray: 4x4 homogeneous transformation matrix T such that (T * [ref_point,1]^T) ≈ [known_point,1]^T.
    """
    P = np.asarray(ref_marker, dtype=float)
    Q = np.asarray(known_marker, dtype=float)
    # Compute centroids of each point set
    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)
    # Center the points around the centroids
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    # Compute cross-covariance matrix
    H = P_centered.T @ Q_centered
    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # Ensure a proper rotation (determinant +1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    # Compute the translation
    t = centroid_Q - R @ centroid_P
    # Build the 4x4 homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

#Point_matrix is 1x3, transformation_matrix is 4x4
def apply_affine_TANSFORM_single_point(point_matrix, transformation_matrix):
    """
    Apply a 4x4 affine transformation to a single 3D point.
    Converts the 3D point to homogeneous coordinates by appending 1, multiplies by the 4x4 matrix,
    and returns the transformed 3D point&#8203;:contentReference[oaicite:2]{index=2}.

    Parameters:
        point_matrix (array-like, shape (1,3) or (3,)): The 3D point to transform.
        transformation_matrix (np.ndarray): A 4x4 transformation matrix (as from compute_rigid_body_transform).

    Returns:
        np.ndarray: A 1x3 array containing the transformed 3D point.
    """
    # Ensure point is a flat array of length 3
    p = np.asarray(point_matrix, dtype=float).reshape(-1)
    # Convert to homogeneous coordinates (append 1)
    p_hom = np.concatenate([p, [1.0]])
    # Apply the transformation matrix
    p_transformed_hom = transformation_matrix @ p_hom
    # Return only the x, y, z components as a 1x3 array
    return p_transformed_hom[:3].reshape(1, 3)
#==============END single point relative to marker===================

def ROM(hip, knee, tibia_knee, ankle):
    # Vectors pointing TOWARD the joint
    # p0 = np.array(knee)
    # d = np.array(hip) - p0
    # femur_vector = d # From hip to knee
           
    # p1 = np.array(ankle)
    # d1 = np.array(tibia_knee) - p0
    # tibia_vector = d1  # From ankle to knee (reversed direction)
    femur_vector = knee - hip        # From hip to knee
    tibia_vector = tibia_knee - ankle
    
    plane_normal = sagittal_normal_femur
    #print("siggtal plane normal in rom: ",plane_normal)
    
    angle = calculate_signed_angle(femur_vector[0], tibia_vector[0], plane_normal)
    angle = abs(angle)
    print("ROM : ",angle)

    #Convert to anatomical angle (0° = full extension, 180° = full flexion)
    anatomical_angle = 180 - abs(angle)
    
    #print(f"Raw Angle: {angle:.1f}° | Anatomical: {anatomical_angle:.1f}°")
    
    return anatomical_angle

def project_point_onto_plane(point, plane_point, plane_normal):
    vector_to_point = point - plane_point
    distance = np.dot(vector_to_point, plane_normal)
    projected_point = point - distance * plane_normal
    return projected_point

def signed_angle(a, b, axis):
    axis = axis / np.linalg.norm(axis)
    a_proj = a - np.dot(a, axis) * axis
    b_proj = b - np.dot(b, axis) * axis
    if np.linalg.norm(a_proj) == 0 or np.linalg.norm(b_proj) == 0:
        return 0.0
    a_proj_norm = a_proj / np.linalg.norm(a_proj)
    b_proj_norm = b_proj / np.linalg.norm(b_proj)
    dot = np.dot(a_proj_norm, b_proj_norm)
    dot = np.clip(dot, -1.0, 1.0)
    angle = np.degrees(np.arccos(dot))
    cross = np.cross(a_proj_norm, b_proj_norm)
    sign = np.sign(np.dot(cross, axis))
    return angle * sign

def Angle(hip, knee, tibia_knee, ankle, coronal_Dir, side):
    # Calculate mechanical axes
    femur_vector = knee - hip
    tibia_vector = tibia_knee - ankle  # From ankle to tibia_knee

    plane_normal = coronal_Dir
    #angle = calculate_signed_angle(femur_vector[0], tibia_vector[0], plane_normal)

    # Compute signed angle between vectors around coronal_Dir
    angleVarVal1 = signed_angle(femur_vector[0],tibia_vector[0], coronal_Dir)
    #print("angleVarVal1:",angleVarVal1)
    displayed_value = 180 - abs(angleVarVal1)
    displayed_value = abs(displayed_value)
    #print(f"Angle: {displayed_value:.0f}")

    # Determine Varus/Valgus based on side and angle sign
    epsilon = 0 #90
    varus = False
    valgus = False
    angle_str = f"{displayed_value:.1f}°"

    if side == "Left":
        if angleVarVal1 > epsilon:
            valgus = True
            angle_str = f"{displayed_value:.0f} Valgus"
        else:
            varus = True
            angle_str = f"{displayed_value:.0f} Varus"
    else:  # Right side
        if angleVarVal1 > epsilon:
            varus = True
            angle_str = f"{displayed_value:.0f} Varus"
        else:
            valgus = True
            angle_str = f"{displayed_value:.0f} Valgus"

    #print(f"Varus/Valgus Angle: {angle_str}")
    return  angle_str, angleVarVal1


def reset_capture_data(obj):
    """
    Reset captured data for tibia or femur, clearing lists and flags.
    """
    global tibia_list, ac_Fetcher_List, femur_list
    
    global tibia_points_done, femur_points_done

    if obj == "tibia":
        tibia_list.clear()
        
        ac_Fetcher_List.clear()
        tibia_points_done = False
        update_current_point("Tibia points reset. Ready for Knee Center ??")
        print("[Reset] Tibia data cleared.")

    elif obj == "femur":
        femur_list.clear()
        
        femur_points_done = False
        update_current_point("Femur points reset. Ready for Hip Center ??")
        print("[Reset] Femur data cleared.")

    else:
        print("[Reset] Unknown object type.")


def perform_reset_o(selected_obj):
    """
    Centralized reset function that works for both keyboard and web
    """
    global femur_list, tibia_list, relative_femur_points_list, relative_tibia_points_list
    global hip_center_fetcher_list, hip_center_fetcher_print_list
    global ac_Fetcher_List, femurCount, tibiaCount, femurFirstFrame
    global is_taking_Hip_Center, OnlyAtFirstTime
    
    print(f"[Reset] Resetting {selected_obj}")
    
    if selected_obj == "femur":
        femurFirstFrame = None
        femur_list = []
        relative_femur_points_list = []
        hip_center_fetcher_list = []
        hip_center_fetcher_print_list = []
        femurCount = None
        is_taking_Hip_Center = True
        OnlyAtFirstTime = True
        
        if WEB_MODE_ENABLED:
            update_point_count('femur', 0)
            update_next_femur_point()  # <-- MAKE SURE THIS IS CALLED
        print("[Reset] Femur points cleared")
        
    elif selected_obj == "tibia":
        # ... (tibia reset logic) ...
        if WEB_MODE_ENABLED:
            update_point_count('tibia', 0)
            update_next_tibia_point()  # <-- Use this for consistency
        print("[Reset] Tibia points cleared")
    
    return True

def perform_reset(selected_obj):
    """
    Centralized reset function that works for both keyboard and web
    """
    global femur_list, tibia_list, relative_femur_points_list, relative_tibia_points_list
    global hip_center_fetcher_list, hip_center_fetcher_print_list
    global ac_Fetcher_List, femurCount, tibiaCount, femurFirstFrame
    global is_taking_Hip_Center, OnlyAtFirstTime
    
    print(f"[Reset] Resetting {selected_obj}")
    
    if selected_obj == "femur":
        femurFirstFrame = None
        femur_list = []
        relative_femur_points_list = []
        hip_center_fetcher_list = []
        hip_center_fetcher_print_list = []
        femurCount = None
        is_taking_Hip_Center = True
        OnlyAtFirstTime = True
        
        if WEB_MODE_ENABLED:
            update_point_count('femur', 0)
            update_next_femur_point() # <-- ADD THIS
        print("[Reset] Femur points cleared")
        
    elif selected_obj == "tibia":
        tibia_list = []
        relative_tibia_points_list = []
        ac_Fetcher_List = []
        tibiaCount = None
        
        if WEB_MODE_ENABLED:
            update_point_count('tibia', 0)
            # update_current_point("Tibia points reset. Ready for Knee Center ??")
            update_next_tibia_point() # <-- Use this for consistency
        print("[Reset] Tibia points cleared")
    
    return True

varAngle_combined = ""
varflex_combined = ""
Tmc_distance = 0
Tlc_distance = 0
Dmp_distance = 0
Dlp_distance = 0


show_reset_text = False
reset_start_time = 0
isRomOn = False
femurFirstFrame = None
selectedObject = None #femur or tibia or none
CT_side = ""  # L for left and R for right
OnlyAtFirstTime = True
CT_side_selected = False 
side_selection_active = True 




def smooth_normal(prev_normal, current_normal, alpha=0.9):
    if prev_normal is None:
        return current_normal
    n = alpha * prev_normal + (1 - alpha) * current_normal
    n /= np.linalg.norm(n)  # keep unit vector
    return n
    
def smooth_point(prev_point, current_point, alpha=0.9):
    if prev_point is None:
        return current_point
    return alpha * prev_point + (1 - alpha) * current_point

def save_tibia_distances_to_xml(tibia_list, malleolus_points, file_path="tibia_distances.xml"):
    """
    Save distances between Knee Center (first tibia point)
    and all other tibia points into XML.
    
    Args:
        tibia_list: List of 6 points [KC, TMC, TLC, Tuberosity, PCL, Ankle Center]
        malleolus_points: List of 2 points [Medial Malleolus, Lateral Malleolus]
        file_path: Path to save XML file
    
    XML will include distances from KC to all points including the malleoli.
    """
    if len(tibia_list) < 2:
        print("[XML] Not enough tibia points to compute distances.")
        return False

    # Root and reference point
    root = ET.Element("TibiaDistances")
    knee_center = np.array(tibia_list[0])

    # Define point names for tibia_list (excluding KC at index 0)
    tibia_point_names = [
        "Tibia Medial Condyle",   # tibia_list[1]
        "Tibia Lateral Condyle",  # tibia_list[2]
        "Tuberosity",             # tibia_list[3]
        "PCL",                    # tibia_list[4]
        "Ankle Center"            # tibia_list[5]
    ]

    # Save distances for regular tibia points
    for i, point in enumerate(tibia_list[1:], start=1):
        point_name = tibia_point_names[i - 1] if (i - 1) < len(tibia_point_names) else f"Point_{i}"
        distance = float(np.linalg.norm(np.array(point) - knee_center))

        point_element = ET.SubElement(root, "Distance")
        point_element.set("from", "KneeCenter")
        point_element.set("to", point_name)
        point_element.text = f"{distance:.2f}"

    # Save distances for malleolus points (if provided)
    if malleolus_points and len(malleolus_points) == 2:
        malleolus_names = ["Medial Malleolus", "Lateral Malleolus"]
        
        for i, point in enumerate(malleolus_points):
            distance = float(np.linalg.norm(np.array(point) - knee_center))
            
            point_element = ET.SubElement(root, "Distance")
            point_element.set("from", "KneeCenter")
            point_element.set("to", malleolus_names[i])
            point_element.text = f"{distance:.2f}"
        
        print(f"[XML] Added malleolus points: {malleolus_names}")

    # Write to XML
    tree = ET.ElementTree(root)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    tree.write(file_path)
    print(f"[XML] Distances saved to {file_path}")
    print(f"[XML] Total points saved: {len(tibia_list) - 1 + (2 if malleolus_points else 0)}")
    return True

def save_femur_distances_to_xml(femur_list, file_path="femur_distances.xml"):
    """
    Save distances between Knee Center (second femur point)
    and all other femur points into XML.
    First point = Hip Center, Second point = Knee Center (reference)
    """
    if len(femur_list) < 2:
        print("[XML] Not enough femur points to compute distances.")
        return False

    # Root and reference (Knee Center)
    root = ET.Element("FemurDistances")
    knee_center = np.array(femur_list[1])

    # Define point names (adjust or extend as needed)
    point_names = [
        "Hip Center",
        "Knee Center",
        "Femur Medial Condyle",
        "Femur Lateral Condyle",
        "Femur Mechanical Axis",
        "Femur Distal Point"
    ]

    # Calculate distances from Knee Center to all other points
    for i, point in enumerate(femur_list):
        if i == 1:
            continue  # Skip knee center itself

        point_name = point_names[i] if i < len(point_names) else f"Point_{i+1}"
        distance = float(np.linalg.norm(np.array(point) - knee_center))

        point_element = ET.SubElement(root, "Distance")
        point_element.set("from", "KneeCenter")
        point_element.set("to", point_name)
        point_element.text = f"{distance:.2f}"

    # Write XML file
    tree = ET.ElementTree(root)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    tree.write(file_path)
    print(f"[XML] Femur distances saved to {file_path}")
    return True

def send_marker_status_to_web(data_instance):
    """
    Extract marker validation status from data_instance and send to web interface.
    This function should be called in the main loop after getting marker data.
    """
    if not WEB_MODE_ENABLED:
        return
    
    try:
        # Check if calibration is complete
        calibrated = data_instance.is_calibrated()
        
        # Get marker IDs
        marker_ids = data_instance.ids  # [8, 5, 11, 13]
        
        # Get latest corner points
        corner_point, _, _ = data_instance.send_data(return_median=True)
        
        # Build marker status dictionary
        markers_data = {}
        
        for idx, marker_id in enumerate(marker_ids):
            if corner_point is not None and idx < len(corner_point):
                marker_corners = corner_point[idx]
                
                # Check if marker is detected
                detected = not np.allclose(marker_corners, 0)
                
                if detected:
                    # Calculate marker size
                    try:
                        from main1 import calculate_distance
                        dists = [
                            calculate_distance(marker_corners[0], marker_corners[1]),
                            calculate_distance(marker_corners[1], marker_corners[2]),
                            calculate_distance(marker_corners[2], marker_corners[3]),
                            calculate_distance(marker_corners[3], marker_corners[0])
                        ]
                        size = np.mean(dists)
                        
                        # Check if valid (only if calibrated)
                        if calibrated and hasattr(data_instance, 'EXPECTED_MARKER_SIZE'):
                            expected = data_instance.EXPECTED_MARKER_SIZE
                            tolerance = data_instance.MARKER_SIZE_TOLERANCE
                            valid = (expected - tolerance <= size <= expected + tolerance)
                        else:
                            # During calibration, all detected markers are shown as valid
                            valid = True
                        
                        markers_data[marker_id] = {
                            'detected': True,
                            'valid': valid,
                            'size': float(size)
                        }
                    except Exception as e:
                        print(f"[Marker Status] Error calculating size for marker {marker_id}: {e}")
                        markers_data[marker_id] = {
                            'detected': True,
                            'valid': False,
                            'size': 0.0
                        }
                else:
                    markers_data[marker_id] = {
                        'detected': False,
                        'valid': False,
                        'size': 0.0
                    }
            else:
                markers_data[marker_id] = {
                    'detected': False,
                    'valid': False,
                    'size': 0.0
                }
        
        # Send to Flask
        update_marker_status(calibrated, markers_data)
        
    except Exception as e:
        print(f"[Marker Status] Error sending to web: {e}")

if __name__ == '__main__':

    # System Configuration
    system_config = {
        "aruco_ids": [8, 5, 11, 13],
        "aruco_size": 50,
        "aruco_dictionary": "DICT_4X4_50"
    }
    
    if selectedObject == "femur":
        is_taking_Hip_Center = True
  
    cv2.setNumThreads(1)
    
    # Initialize data and camera systems
    data_instance_1 = data_1()
    marker_size = system_config['aruco_size']

    # Load pre-computed data
    virtual_sq = SQ_MARKER_NAVIGATION_SUPPORT_LIB.get_vertual_SQ_coordinates(marker_size)
    pivot_data = np.load("pivot_point_cam.npy", allow_pickle=True).item()
    pivot_point_cam = pivot_data["pivot_tip_point"]
    pivot_direction = pivot_data["pivot_direction"]
    
    pivot_def = np.vstack((virtual_sq, pivot_point_cam))
    pivot_point_obj = SQ_MARKER_PIVOT_OBJ(pivot_def)
    
    # Pre-load ArUco dictionary
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    # Initialize tracking variables
    T_ref_tib = None
    T_ref_fem = None
    tibia_list = []
    femur_list = []
    relative_tibia_points_list = []
    relative_femur_points_list = []
    tibiaCount = None
    femurCount = None
    
    # Initialize state management
    state = {
        'CT_side': None,
        'CT_side_selected': False,
        'femur_first_frame': None,
        'show_reset_text': False,
        'reset_start_time': None,
        'last_trigger_time': 0,
        'last_w_key_time': 0,
        'last_ct_side_time': 0,
        'rom_angle': 0,
        'coronal_rom_angle': 0
    }
    
    # Timing constants (milliseconds)
    COOLDOWN_DURATION = 1000
    W_KEY_DEBOUNCE = 100
    CT_SIDE_DEBOUNCE = 200
    KEY_CHECK_DELAY = 0.01
    
    # UI configuration
    RIGHT_TEXT_X = 430
    RIGHT_TEXT_Y = 80
    
    zero_arr = np.zeros((4, 3))
    prev_vp_normal = None
    prev_center = None
    
    print(system_config)
    print("virtual_sq", virtual_sq)
    print("pivot_point_cam:", pivot_point_cam)
    print(f"Web mode: {'ENABLED' if WEB_MODE_ENABLED else 'DISABLED'}")
    
    target_matrix = target_matrix.T
    frame_count = 0
    
    # Complete main loop with proper capture control

    while True:
        current_time = time.time() * 1000  # Milliseconds
        
        # ==================== GET CAMERA FRAMES ====================
        ret, left, right, is_marker_detected = data_instance_1.obj2.get_IR_FRAME_SET()
        
        if not ret:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Closing application...")
                on_closing()
                break
            continue
        
        # ==================== ARUCO DETECTION ====================
        corners, ids, rejected = cv2.aruco.detectMarkers(left, dictionary, [8, 5, 11, 13])
        if corners:
            cv2.aruco.drawDetectedMarkers(left, corners, ids, borderColor=(255, 255, 255))
            for corner in corners:
                pts = corner[0].astype(int)
                cv2.polylines(left, [pts], isClosed=True, color=(255, 255, 255), thickness=4)
        
        # Get marker corner points
        corner_point, center_points, is_marker_detected = data_instance_1.send_data(return_median=True)
        send_marker_status_to_web(data_instance_1)
        exact_matches = [np.array_equal(arr, zero_arr) for arr in corner_point]
        
        # ==================== UPDATE REFERENCE TRANSFORMS ====================
        if selectedObject == "tibia":
            aruco_tibiacorners_ref = corner_point[2]
            T_ref_tib = SQ_MARKER_NAVIGATION_SUPPORT_LIB.GET_ARUCO_VREF_TRANSFORM(aruco_tibiacorners_ref, marker_size)
        elif selectedObject == "femur":
            aruco_femurcorners_ref = corner_point[3]
            T_ref_fem = SQ_MARKER_NAVIGATION_SUPPORT_LIB.GET_ARUCO_VREF_TRANSFORM(aruco_femurcorners_ref, marker_size)
        
        # ==================== CALCULATE VERIFICATION PLANE NORMALS ====================
        if not (corner_point[1] == zero_arr).all():
            if selectedObject == "tibia":
                vp_corners = getRelativeToMarker(corner_point[1], corner_point[2])
                center_points[1] = np.mean(vp_corners, axis=0)
                vp_normal = Planes_genrate_LIB.calculate_plane_normal(vp_corners)
            elif selectedObject == "femur":
                vp_corners = getRelativeToMarker(corner_point[1], corner_point[3])
                center_points[1] = np.mean(vp_corners, axis=0)
                vp_normal = Planes_genrate_LIB.calculate_plane_normal(vp_corners)
        
        # ==================== WEB COMMANDS ====================
        if WEB_MODE_ENABLED:
            cmd_state = get_command_state()
            
            # Update selected object from web
            if cmd_state['selected_object'] != selectedObject:
                selectedObject = cmd_state['selected_object']
                print(f"[Web] Object changed to: {selectedObject}")
            
            # Update CT side from web
            if cmd_state['ct_side'] and not state['CT_side_selected']:
                state['CT_side'] = cmd_state['ct_side']
                CT_side = cmd_state['ct_side']
                state['CT_side_selected'] = True
                print(f"[Web] CT Side set to: {CT_side}")
        
        # ==================== KEYBOARD INPUT HANDLING ====================
        key = cv2.waitKey(1) & 0xFF
        
        # Flags for capture control
        should_capture_point = False
        should_finalize_hip = False
        capture_source = None
        
        if key != 0xFF:  # Key pressed
            # ===== CT SIDE SELECTION =====
            if not state['CT_side_selected']:
                if key == ord('l') and (current_time - state['last_ct_side_time']) > CT_SIDE_DEBOUNCE:
                    state['CT_side'] = "L"
                    CT_side = "L"
                    state['CT_side_selected'] = True
                    state['last_ct_side_time'] = current_time
                    print("[Keyboard] CT Side: L")
                elif key == ord('r') and (current_time - state['last_ct_side_time']) > CT_SIDE_DEBOUNCE:
                    state['CT_side'] = "R"
                    CT_side = "R"
                    state['CT_side_selected'] = True
                    state['last_ct_side_time'] = current_time
                    print("[Keyboard] CT Side: R")
            
            # ===== OBJECT SELECTION =====
            if key == ord('f'):
                selectedObject = "femur"
                isRomOn = False
                print("[Keyboard] Selected: Femur")
                if WEB_MODE_ENABLED: update_next_femur_point()
            elif key == ord('t'):
                selectedObject = "tibia"
                isRomOn = False
                print("[Keyboard] Selected: Tibia")
                if WEB_MODE_ENABLED: update_next_tibia_point()
            
            elif key == ord('r'):
                state['show_reset_text'] = True
                state['reset_start_time'] = time.time()
                perform_reset(selectedObject)  # Use centralized function
                
            # ===== POINT REMOVAL =====
            elif key == ord('m') and (current_time - state['last_trigger_time']) > COOLDOWN_DURATION:
                state['last_trigger_time'] = current_time
                
                if selectedObject == "femur":
                    if len(femur_list) == 1:
                        state['femur_first_frame'] = None
                        hip_center_fetcher_list = []
                        hip_center_fetcher_print_list = []
                        is_taking_Hip_Center = True
                        femur_list = []
                        relative_femur_points_list = []
                        OnlyAtFirstTime = True
                    elif 0 < len(femur_list) < 6 and not is_taking_Hip_Center:
                        femur_list.pop()
                        relative_femur_points_list.pop()
                    elif len(femur_list) == 0 and len(hip_center_fetcher_list) > 0:
                        hip_center_fetcher_list.pop()
                        hip_center_fetcher_print_list.pop()
                    
                    if WEB_MODE_ENABLED:
                        update_point_count('femur', len(femur_list))
                    print(f"[Remove] Femur point removed. Total: {len(femur_list)}")
                
                elif selectedObject == "tibia":
                    if 0 < len(tibia_list) < 6 and len(ac_Fetcher_List) == 0:
                        tibia_list.pop()
                        relative_tibia_points_list.pop()
                    elif len(ac_Fetcher_List) > 0:
                        ac_Fetcher_List.pop()
                    
                    if WEB_MODE_ENABLED:
                        update_point_count('tibia', len(tibia_list))
                    print(f"[Remove] Tibia point removed. Total: {len(tibia_list)}")
            
            # ===== QUIT =====
            elif key == ord('q'):
                print("Closing application...")
                on_closing()
                break
        
        # ==================== WEB CAPTURE TRIGGER ====================
        if WEB_MODE_ENABLED and cmd_state.get('capture_point'):
            should_capture_point = True
            capture_source = "web"
            reset_capture_flag()
            
        # ==================== WEB RESET TRIGGER ====================
        if WEB_MODE_ENABLED and cmd_state.get('reset_triggered'):
            print(f"[Web] Reset triggered for {selectedObject}")
            perform_reset(selectedObject)
            reset_reset_flag()  # Clear the flag
            show_reset_text = True
            reset_start_time = time.time()
        
        # ==================== FINALIZE HIP CENTER ====================
        #if should_finalize_hip:
        #    if finalize_hip_center():
        #        if WEB_MODE_ENABLED:
        #            update_point_count('femur', len(femur_list))
        
        # ==================== EXECUTE POINT CAPTURE ====================
        if should_capture_point:
            print(f"[Capture] Triggered by {capture_source} for {selectedObject}")
            
            if selectedObject == "tibia":
                if not (corner_point[0] == zero_arr).all() and not (corner_point[2] == zero_arr).all() and T_ref_tib is not None:
                    success = tibia_pointtaker()
                    if success:
                        if WEB_MODE_ENABLED:
                            update_point_count('tibia', len(tibia_list))
                        print(f"[Success] Tibia point {len(tibia_list)}/6 captured")
                    else:
                        print("[Failed] Could not capture tibia point")
                else:
                    print("[Failed] Markers not detected or reference not set")
            
            elif selectedObject == "femur":
                if not (corner_point[3] == zero_arr).all() and T_ref_fem is not None:
                    success = femur_pointtaker()
                    if success:
                        if WEB_MODE_ENABLED:
                            if is_taking_Hip_Center:
                                # Still collecting hip center points
                                pass
                            else:
                                update_point_count('femur', len(femur_list))
                        print(f"[Success] Femur point captured")
                    else:
                        print("[Failed] Could not capture femur point")
                else:
                    print("[Failed] Markers not detected or reference not set")
        
        # ==================== VERIFICATION (runs every frame) ====================
        if selectedObject == "tibia":
            tibia_verifycuts()
            tibia_point_referencing()
        elif selectedObject == "femur":
            femur_verifycuts()
            femur_point_referencing()
        
        # ==================== RESET TEXT DISPLAY ====================
        if state['show_reset_text'] and state['reset_start_time']:
            elapsed_time = time.time() - state['reset_start_time']
            if elapsed_time < 2:
                reset_msg = f"Reset {selectedObject.capitalize()} Points!"
                cv2.putText(left, reset_msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                state['show_reset_text'] = False
        
        # ==================== ROM CALCULATION ====================
        if isRomOn and corner_point[2] is not None and corner_point[3] is not None:
            try:
                new_femur_hip = virtual_to_camera(relative_femur_points_list[0], corner_point[3])
                new_femur_knee = virtual_to_camera(relative_femur_points_list[1], corner_point[3])
                new_tibia_knee = virtual_to_camera(relative_tibia_points_list[0], corner_point[2])
                new_tibia_ankle = virtual_to_camera(relative_tibia_points_list[3], corner_point[2])
                
                new_femur_hip = getRelativeToPoint(new_femur_hip, corner_point[3])
                new_femur_knee = getRelativeToPoint(new_femur_knee, corner_point[3])
                new_tibia_knee = getRelativeToPoint(new_tibia_knee, corner_point[3])
                new_tibia_ankle = getRelativeToPoint(new_tibia_ankle, corner_point[3])
                
                state['rom_angle'] = ROM(new_femur_hip, new_femur_knee, new_tibia_knee, new_tibia_ankle)
                state['coronal_rom_angle'] = Angle(new_femur_hip, new_femur_knee, new_tibia_knee,
                                                    new_tibia_ankle, sagittal_normal_femur, state['CT_side'])[0]
            except Exception as e:
                print(f"ROM Calculation Error: {str(e)}")
        
        # ==================== UI TEXT DISPLAY ====================
        if selectedObject == "femur":
            cv2.putText(left, "Femur", (RIGHT_TEXT_X, RIGHT_TEXT_Y - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            msg = ""
            # Check states in the correct order, using the global variables
            if femurFirstFrame is None:
                # 1. Waiting for First Frame
                msg = "Ready for Femur First Frame"
                
            elif is_taking_Hip_Center:
                # 2. Collecting Hip Center Points
                count = len(hip_center_fetcher_list)
                if count < 5:
                    msg = f"Hip Center {count + 1}/5"
                else:
                    msg = "Finalizing Hip Center..." # This state is brief
                    
            else:
                # 3. Collecting Anatomical Points (Hip Center is done)
                count = len(femur_list) # Will be 1 (HC) up to 6
                
                femur_anatomical_messages = {
                    1: " Femur Knee Center",
                    2: " Distal Medial Condyle",
                    3: " Distal Lateral Condyle",
                    4: " Surgical Medial",
                    5: " Surgical Lateral",
                    6: " Femur Points Complete"
                }
                
                # Get the message based on the count, default to "Complete" if > 6
                msg = femur_anatomical_messages.get(count, "Femur Points Complete")

            # Draw the determined message
            cv2.putText(left, msg, (RIGHT_TEXT_X, RIGHT_TEXT_Y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        elif selectedObject == "tibia":
            cv2.putText(left, "Tibia", (RIGHT_TEXT_X, RIGHT_TEXT_Y - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            tibia_messages = [
                "Tibia Knee Center",
                "Tibia Medial Condyle",
                "Tibia Lateral Condyle",
                "Tuberosity",
                "PCL",
                "Medial Malleolus" if len(ac_Fetcher_List) == 0 else 
                ("Lateral Malleolus" if len(ac_Fetcher_List) == 1 else "Ankle Center Complete"),
                "Tibia Points Complete"
            ]
            
            if len(tibia_list) < len(tibia_messages):
                msg = tibia_messages[len(tibia_list)]
                cv2.putText(left, msg, (RIGHT_TEXT_X, RIGHT_TEXT_Y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Display CT side
        #if state['CT_side']:
        #    cv2.putText(left, state['CT_side'], (RIGHT_TEXT_X + 80, RIGHT_TEXT_Y - 50),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Display marker detection status
        marker_status = "✓ All Markers Detected" if all(not np.array_equal(arr, zero_arr) for arr in corner_point[:4]) else "⚠ Markers Missing"
        cv2.putText(left, marker_status, (10, left.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if "✓" in marker_status else (0, 0, 255), 2)
        
        # Display capture instruction
        cv2.putText(left, "Press 'c' or click 'Capture Point' to capture", (10, left.shape[0] - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # ==================== DISPLAY FRAME ====================
        cv2.imshow("Camera Frame left", left)
        
        # ==================== UPDATE WEB FRAME ====================
        if WEB_MODE_ENABLED:
            update_frame(left)
        
        # ==================== LOOP DELAY ====================
        time.sleep(0.05)
