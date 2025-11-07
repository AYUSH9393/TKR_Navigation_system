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
from main1 import data
from CAMERA_CAPTURE_MODULE_MAIN import ComeraCapture_RealSense
from collections import deque
import SQ_MARKER_NAVIGATION_SUPPORT_LIB
from SQ_MARKER_NAVIGATION_SUPPORT_LIB import SQ_MARKER_PIVOT_OBJ
import REG_AND_TRANS_SUPPORT_LIB
import Planes_genrate_LIB
import pyvista as pv
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
import re
#nik
#socket related Code & variables
#---------------------------------
# Global references
camera_on_img = None
camera_off_img = None
camera_icon_label = None
# Connect to HoloLens IP and port
host = ""  # Replace with your HoloLens IP
PORT = 9980
#HOLOLENS_IP = input("Enter HoloLens IP: ")

def is_valid_ip(ip):
    """Validate IP address format using regex"""
    pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    if not re.match(pattern, ip):
        return False
    return all(0 <= int(octet) <= 255 for octet in ip.split('.'))


def check_ip_connection(ip, port=5000, timeout=3):
    """
    Tries to connect to the given IP and port.
    Returns True if successful, False otherwise.
    """
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except (socket.timeout, socket.error):
        return False



def get_ip_address():
    """
    Creates a simple dialog window to get the HoloLens IP address
    Returns the IP address as string
    """
    ip_window = tk.Tk()
    ip_window.title("HoloLens IP Configuration")
    
    # Center the window
    window_width = 400
    window_height = 200
    screen_width = ip_window.winfo_screenwidth()
    screen_height = ip_window.winfo_screenheight()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    ip_window.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
    # Style configurations
    ip_window.configure(bg='black')
    
    # Create and pack widgets
    title_label = tk.Label(
        ip_window, 
        text="Enter HoloLens IP Address",
        font=("Arial", 16, "bold"),
        bg='black',
        fg='white',
        pady=20
    )
    title_label.pack()

    ip_var = tk.StringVar()
    ip_entry = tk.Entry(
        ip_window,
        textvariable=ip_var,
        font=("Arial", 14),
        width=20,
        justify='center'
    )
    ip_entry.pack(pady=10)

    def submit_ip():
        ip = ip_var.get()
        
        if not is_valid_ip(ip):
            messagebox.showerror("Invalid IP", "The IP address format is incorrect.")
            return
        
        if not check_ip_connection(ip, port=9980):
            messagebox.showerror("Connection Failed", f"Cannot connect to {ip}:9980.")
            return

        ip_window.quit()
        ip_window.destroy()
    
    submit_button = tk.Button(
        ip_window,
        text="Connect",
        command=submit_ip,
        font=("Arial", 12),
        bg='#314d79',
        fg='white',
        width=15,
        height=1
    )
    submit_button.pack(pady=20)

    # Add validation message label
    validation_label = tk.Label(
        ip_window,
        text="",
        font=("Arial", 10),
        bg='black',
        fg='red'
    )
    validation_label.pack(pady=5)

    ip_window.mainloop()
    return ip_var.get()

pointCount = 0
messageID = ''
corner_point = []
zeroArr = np.zeros((4,3))
config = configparser.ConfigParser()
config.read('config.ini')

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

def tibia_pointtaker():
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
            pygame.event.pump()  # Poll joystick events
            current_time = pygame.time.get_ticks()

            # Get current button state
            button_a_current = joystick.get_button(0)

            # Detect rising edge (transition from not pressed to pressed)
            button_a_pressed = button_a_current and not prev_button_a
            prev_button_a = button_a_current  

            # Check trigger conditions
            #key_pressed = (key == ord('c'))
            button_allowed = button_a_pressed and (current_time - last_button_a_trigger >= 1000)
            
            if (button_allowed) and len(tibia_list) <= 4 and CT_side_selected:
                # Update cooldown if triggered by button
                if button_a_pressed:
                    last_button_a_trigger = current_time

                if len(tibia_list) == 0:
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
                elif len(tibia_list) == 1:
                    print("TMC")
                    tibia_list.append(pivot_ref)
                    # Get relative position
                    relative_pos = get_relative_position(corner_point[2], pivot_ref)
                    relative_tibia_points_list.append(relative_pos)
                    #tibia_list.append(center_points[0])
                    
                    #draw_a_plot_pointerFourVal(pivot_ref,corner_point[0])
                    print(f"Stored pivot_ref: {pivot_ref}")
                    print(f"Current list: {tibia_list}")
                elif len(tibia_list) == 2:
                    print("TLC")
                    tibia_list.append(pivot_ref)
                    # Get relative position
                    relative_pos = get_relative_position(corner_point[2], pivot_ref)
                    relative_tibia_points_list.append(relative_pos)
                    #draw_a_plot_pointerFourVal(pivot_ref,corner_point[0])
                    #tibia_list.append(center_points[0])
                    print(f"Stored pivot_ref: {pivot_ref}")
                    print(f"Current list: {tibia_list}")
                else:
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

global tibia_list , femur_list

def tibia_verifycuts():
    global tibia_list,tibiaCount,Sig_Dir,ax,drawPlot,Sig_Dir,var,varAngle_combined,varflex_combined,Tmc_distance,Tlc_distance,CT_side
 
    #global coronal_Dir
    # Only proceed if tibia_list has at least 2 elements
    if len(tibia_list) >= 4 and not (corner_point[1] == zeroArr).all():
        # line Ankle to KC
        p0, Dir = Planes_genrate_LIB.create_line(tibia_list[0], tibia_list[3])
        generate_orthogonal_planes, normals = Planes_genrate_LIB.generate_orthogonal_planes((p0, Dir), tibia_list[0])

        # siggital from projecting point on axial plane
        axial_plane = generate_orthogonal_planes['axial']
        if tibiaCount is None:
            axial_plane = Plane(tuple(tibia_list[0]), normal_vector = Dir)
            sagittal_plane, angle,Sig_Dir = Planes_genrate_LIB.generate_sagittal_plane_with_projection([tibia_list[0], tibia_list[1],tibia_list[2]], axial_plane,CT_side)
            angle = 90 - angle
            tibiaCount = 1

        #coronal 
        coronal_Dir = np.cross(Dir,Sig_Dir)
        coronal_plane = (Planes_genrate_LIB.plane_from_point_and_normal(tibia_list[0],coronal_Dir))
        #print("Coronal Plane: ", (coronal_plane))
        Angle_axial_cor = Planes_genrate_LIB.angle_between_planes(axial_plane,coronal_plane)
        #print("Angle between axial and coronal", Angle_axial_cor)

        # varus_valgus
        TibiamechanicalAxis_Normal = Planes_genrate_LIB.project_on_plane(Dir,coronal_Dir)
        TibiamechanicalAxis_Minus_Normal = Planes_genrate_LIB.project_on_plane(-Dir,coronal_Dir)
        TibiaMarker_Normal = Planes_genrate_LIB.project_on_plane(vp_normal,coronal_Dir)
        TibiaMarker_Minus_Normal = Planes_genrate_LIB.project_on_plane(-vp_normal,coronal_Dir)
        varAngle = Planes_genrate_LIB.signed_angle((TibiamechanicalAxis_Normal-TibiamechanicalAxis_Minus_Normal),(TibiaMarker_Normal-TibiaMarker_Minus_Normal),coronal_Dir)[0]
        

        #Ante/Post
        TibiamechanicalAxis_Normal = Planes_genrate_LIB.project_on_plane(Dir,Sig_Dir)
        TibiamechanicalAxis_Minus_Normal = Planes_genrate_LIB.project_on_plane(-Dir,Sig_Dir)
        TibiaMarker_Normal = Planes_genrate_LIB.project_on_plane(vp_normal,Sig_Dir) 
        TibiaMarker_Minus_Normal = Planes_genrate_LIB.project_on_plane(-vp_normal,Sig_Dir)
        varflex = Planes_genrate_LIB.signed_angle((TibiamechanicalAxis_Normal-TibiamechanicalAxis_Minus_Normal),(TibiaMarker_Normal-TibiaMarker_Minus_Normal),Sig_Dir)[0]
        #varFlex = Planes_genrate_LIB.signed_angle((TibiamechanicalAxis_Normal-TibiamechanicalAxis_Minus_Normal),(TibiaMarker_Normal-TibiaMarker_Minus_Normal),Sig_Dir)[1] 
        #print(" TIBiA Ant Angle signed: ",varflex)
        #print(" TIBiA Flex Angle : ",varFlex)
        
        Tmc_distance = -int(Planes_genrate_LIB.point_plane_distance(tibia_list[1],vp_normal,center_points[1]))
        #print("TMC_distance before:",Tmc_distance)
        #Tmc_distance = round(Tmc_distance, 0) 
        #print("TMC_distance:",Tmc_distance)
        Tlc_distance = -int(Planes_genrate_LIB.point_plane_distance(tibia_list[2],vp_normal,center_points[1]))
        #Tlc_distance = round(Tlc_distance, 0)
        #print("TLC_distance:",Tlc_distance)

        if varAngle < 0:
            varAngle_name = "Val" if CT_side == "L" else "Val"
        else:
            varAngle_name = "Var" if CT_side == "L" else "Var"

        # if varflex < 0:
        #     varflex_name = "Ant" if CT_side == "L" else "Ant"
        # else:
        #     varflex_name = "Post" if CT_side == "L" else "Post"

        if CT_side == "L":
            display_angle = -varflex  # Invert for left side
        else:
            display_angle = varflex

        # Display angle with consistent terminology
        if display_angle < 0:
            varflex_name = "Ant"
        else:
            varflex_name = "Post"
                
        # Combine names and values into formatted strings
        varAngle_combined = f" {abs(varAngle):.0f} {varAngle_name}"
        varflex_combined = f" {abs(display_angle):.0f} {varflex_name}"

        # Encode the combined strings to fixed-length byte strings
        varAngle_encoded = varAngle_combined.encode('utf-8').ljust(32, b'\x00')  # 32 bytes
        varflex_encoded = varflex_combined.encode('utf-8').ljust(32, b'\x00')    # 32 bytes

        #Pack the buffer with the new fields
        buffer = struct.pack(
            '16s8s32s32sff',
            selected_object_encoded,
            ct_side_encoded,
            varAngle_encoded,
            varflex_encoded,
            Tlc_distance,
            Tmc_distance,
        )

        # Send buffer
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            #s.connect((getTheIP(), PORT))
            s.connect((HOLOLENS_IP, PORT))
            s.sendall(buffer)
            print("Data sent tibia.")
        #print(tibia_list)
        
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


removedFirstValue = False
hip_center_fetcher_list = []
hip_center_fetcher_print_list = []
is_taking_Hip_Center = True
prev_button_a_femur = False
last_button_a_trigger_femur = 0

def femur_pointtaker():
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
        pygame.event.pump()  # Poll joystick events
        current_time = pygame.time.get_ticks()

        # Get current button state
        button_a_current = joystick.get_button(0)

        # Detect rising edge (transition from not pressed to pressed)
        button_a_pressed = button_a_current and not prev_button_a_femur
        prev_button_a_femur = button_a_current  # Update previous state

        # Check trigger conditions
        key_pressed = (key == ord('c'))
        button_allowed = button_a_pressed and (current_time - last_button_a_trigger_femur >= 1000)
        
        if (key_pressed or button_allowed) and femurFirstFrame is not None and len(femur_list) <= 4  and CT_side_selected:
            # Update cooldown if triggered by button
            if button_a_pressed:
                last_button_a_trigger_femur = current_time
                
            print("Pressed C")
            if len(femur_list) == 0:
                print("HC")
                ###HC Finder with 5 points.
                if not (corner_point[3] == zeroArr).all() and not (corner_point[1] == zeroArr).all():
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

    except Exception as e:
        print(f"Error computing pivot point: {e.with_traceback()}")

def femur_verifycuts():
    global femur_list,femurCount,Sig_Dir_femur,ax,drawPlot,coronal_Dir,sagittal_plane,Dir,Dmp_distance,Dlp_distance,varAngle_combined,varflex_combined
    
    # Only proceed if tibia_list has at least 2 elements
    if len(femur_list) >= 4 and not (corner_point[1] == zeroArr).all():
        # line hipCenter to KC
        p0, Dir = Planes_genrate_LIB.create_line(femur_list[1], femur_list[0])
        generate_orthogonal_planes, normals = Planes_genrate_LIB.generate_orthogonal_planes((p0, Dir), femur_list[1])

        # siggital from projecting point on axial plane
        axial_plane = generate_orthogonal_planes['axial']
        if femurCount is None:
            axial_plane = Plane(tuple(femur_list[1]), normal_vector = Dir)

            sagittal_plane, angle,Sig_Dir_femur = Planes_genrate_LIB.generate_sagittal_plane_with_projection([femur_list[1], femur_list[2],femur_list[3]], axial_plane,CT_side)
            angle = 90 - angle
            femurCount = 1

        #coronal 
        #print("siggtal plane normal in verify: ",Sig_Dir_femur)
        coronal_Dir = np.cross(Dir,Sig_Dir_femur)
        coronal_plane = (Planes_genrate_LIB.plane_from_point_and_normal(femur_list[1],coronal_Dir))
        #print("Coronal Plane: ", (coronal_plane))
        Angle_axial_cor = Planes_genrate_LIB.angle_between_planes(axial_plane,coronal_plane)
        Angle_sig_cor = Planes_genrate_LIB.angle_between_planes(sagittal_plane,coronal_plane)
        #print("Angle between sagital and coronal", Angle_sig_cor)
        #print("Angle between axial and coronal", Angle_axial_cor)

        # varus_valgus
        FemurmechanicalAxis_Normal = Planes_genrate_LIB.project_on_plane(Dir,coronal_Dir)
        FemurmechanicalAxis_Minus_Normal = Planes_genrate_LIB.project_on_plane(-Dir,coronal_Dir)
        FemurMarker_Normal = Planes_genrate_LIB.project_on_plane(vp_normal,coronal_Dir)
        FemurMarker_Minus_Normal = Planes_genrate_LIB.project_on_plane(-vp_normal,coronal_Dir)
        varAngle = Planes_genrate_LIB.signed_angle((FemurmechanicalAxis_Normal-FemurmechanicalAxis_Minus_Normal),(FemurMarker_Normal-FemurMarker_Minus_Normal),coronal_Dir)[0]
       
        #Ante/Post
        FemurmechanicalAxis_Normal = Planes_genrate_LIB.project_on_plane(Dir,Sig_Dir_femur)
        FemurmechanicalAxis_Minus_Normal = Planes_genrate_LIB.project_on_plane(-Dir,Sig_Dir_femur)
        FemurMarker_Normal = Planes_genrate_LIB.project_on_plane(vp_normal,Sig_Dir_femur)
        FemurMarker_Minus_Normal = Planes_genrate_LIB.project_on_plane(-vp_normal,Sig_Dir_femur)
        varflex = Planes_genrate_LIB.signed_angle((FemurmechanicalAxis_Normal-FemurmechanicalAxis_Minus_Normal),(FemurMarker_Normal-FemurMarker_Minus_Normal),Sig_Dir_femur)[0]
        #print("femur varflex angle: ",varflex)
       
        Dmp_distance = -int(Planes_genrate_LIB.point_plane_distance(femur_list[2],vp_normal,center_points[1]))
        Dlp_distance = -int(Planes_genrate_LIB.point_plane_distance(femur_list[3],vp_normal,center_points[1]))


        if varAngle < 0:
            varAngle_name = "Var" if CT_side == "L" else "Var"
        else:
            varAngle_name = "Val" if CT_side == "L" else "Val"

        # if varflex < 0:
        #     varflex_name = "ext" if CT_side == "L" else "ext"
        # else:
        #     varflex_name = "flex" if CT_side == "L" else "flex"

        if CT_side == "L":
            display_angle = -varflex  # Invert for left side
        else:
            display_angle = varflex

        if display_angle < 0:
            varflex_name = "ext"
        else:
            varflex_name = "flex"

        # Combine names and values into formatted strings
        varAngle_combined = f"{abs(varAngle):.0f} {varAngle_name}"
        varflex_combined = f" {abs(varflex):.0f} {varflex_name}"

        # Encode the combined strings to fixed-length byte strings
        varAngle_encoded = varAngle_combined.encode('utf-8').ljust(32, b'\x00')  # 32 bytes
        varflex_encoded = varflex_combined.encode('utf-8').ljust(32, b'\x00')    # 32 bytes

        # Pack the buffer with the new fields
        buffer = struct.pack(
            '16s8s32s32sff',
            selected_object_encoded,
            ct_side_encoded,
            varAngle_encoded,
            varflex_encoded,
            Dlp_distance,
            Dmp_distance,
        )

        # Send buffer
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            #s.connect((getTheIP(), PORT))
            s.connect((HOLOLENS_IP, PORT))
            s.sendall(buffer)
            print("Data sent femur.")
        

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
    
    plane_normal = Sig_Dir_femur
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


varAngle_combined = ""
varflex_combined = ""
Tmc_distance = 0
Tlc_distance = 0
Dmp_distance = 0
Dlp_distance = 0
CT_side = ""
selectedObject = "tibia"


def toggle_camera_display():
    """Toggle camera feed display"""
    global show_camera, camera_icon_label

    show_camera.set(not show_camera.get())

    # Switch icon based on state
    if show_camera.get():
        camera_icon_label.configure(image=camera_on_img)
    else:
        camera_icon_label.configure(image=camera_off_img)
        cam_canvas.configure(image='')
        cam_canvas.image = None


def create_gui():
    """Main GUI using the persistent root window"""
    # root.deiconify()  # Show main window
    # root.title("Main Application")
    
    global cam_canvas
    
    """Create and initialize GUI elements"""
    global root, side_label, object_label, tibia_label, pointer_label
    global verification_label, coronal_label, sagittal_label, medial_label, lateral_label, Point_Taken_label,rom_label,screen_width,screen_height
    
    # Create the main Tkinter window
    root = tk.Tk()
    root.title("Depth of Cut GUI")
    # Get screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    # Set window size to screen size
    root.geometry(f"{screen_width}x{screen_height}")
    root.attributes('-fullscreen', True)
    root.state('zoomed')
    # root.geometry("1920x1080")  # Set window size
    root.configure(bg='black')  # Set main window background to black

    # Create a frame for displaying the selected side and object
    info_frame = tk.Frame(root, bd=2, relief="solid", padx=10, pady=50, bg='black')
    info_frame.pack(fill="x",pady=5)

    # Create main left frame for markers and tracking info
    left_frame = tk.Frame(root, relief="solid", padx=10, bg='black')
    left_frame.pack(side=tk.LEFT, fill=tk.Y, pady=10, padx=50)

    # Display the selected side
    normal_font = tkFont.Font(family="Arial", size=18, weight=tkFont.NORMAL)
    side_label = tk.Label(info_frame, text=f"Patient Side: {CT_side}", 
                         font=normal_font,anchor="w",  padx=screen_width/2.5, pady=-20,
                         bg='black', fg='white')  # Black background, white text
    side_label.pack(fill="x",pady=5)

    # Display the selected object
    object_label = tk.Label(info_frame, text=f"Selected Object: {selectedObject}", 
                          font=("Helvetica", 18), anchor="w", padx=screen_width/2.5, pady=-20,
                          bg='black', fg='white')
    object_label.pack(fill="x", pady=5)

    # Create a frame for displaying marker status
    marker_frame = tk.Frame(left_frame, bd=2, relief="solid", padx=10, pady=5, bg='black')
    marker_frame.pack(fill="x", pady=5)

    # Create labels for markers with black background
    tibia_label = tk.Label(marker_frame, text="", font=("Helvetica", 18), 
                          width=30, height=2, bg='black', fg='white')
    tibia_label.pack(pady=5)

    pointer_label = tk.Label(marker_frame, text="", font=("Helvetica", 18), 
                           width=30, height=2, bg='black', fg='white')
    pointer_label.pack(pady=5)

    verification_label = tk.Label(marker_frame, text="", font=("Helvetica", 18), 
                                width=30, height=2, bg='black', fg='white')
    verification_label.pack(pady=5)

    # Create a frame for point taking status
    point_frame = tk.Frame(left_frame, padx=10, pady=10, bg='black')
    point_frame.pack(pady=5)

    Point_Taken_label = tk.Label(point_frame, text="Point taken", 
                                font=("Helvetica", 18), anchor="w",
                                bg='#314d79', fg='white')
    Point_Taken_label.pack(pady=5)

    # Create right frame for values
    right_frame = tk.Frame(root, bg='black')
    right_frame.pack(side=tk.RIGHT, fill=tk.Y, pady=10, padx=150)

    # Create labels for Coronal, Sagittal, Medial, and Lateral values with uniform width
    label_width = 15  # Set uniform width for all labels
    label_height = 1  # Set uniform height for all labels
    
    coronal_label = tk.Label(right_frame, 
                            text="Coronal: N/A", 
                            font=("Helvetica", 20),
                            anchor="w",
                            width=label_width,
                            height=label_height,
                            bg='#314d79',
                            fg='white')
    coronal_label.pack()

    sagittal_label = tk.Label(right_frame, 
                             text="Sagittal: N/A",
                             font=("Helvetica", 20),
                             anchor="w",
                             width=label_width,
                             height=label_height,
                             bg='#314d79',
                             fg='white')
    sagittal_label.pack()

    medial_label = tk.Label(right_frame, 
                           text="Medial: N/A", 
                           font=("Helvetica", 20),
                           anchor="w",
                           width=label_width,
                           height=label_height,
                           bg='#314d79',
                           fg='white')
    medial_label.pack()

    lateral_label = tk.Label(right_frame, 
                            text="Lateral: N/A", 
                            font=("Helvetica", 20),
                            anchor="w",
                            width=label_width,
                            height=label_height,
                            bg='#314d79',
                            fg='white')
    lateral_label.pack()


    # ROM label with matching width
    rom_label = tk.Label(right_frame, 
                        text="ROM: N/A", 
                        font=("Helvetica", 20),
                        anchor="w",
                        width=label_width,
                        height=label_height,
                        bg='#314d79',
                        fg='white')
    rom_label.pack(pady=5, padx=5, fill="x")

    # Create frame for camera display at the bottom
    camera_frame = tk.Frame(root, bd=2, relief="solid", pady=5, bg='black')
    camera_frame.pack(side=tk.BOTTOM,fill="both", expand=True)

    # Create a frame for camera toggle controls
    camera_control_frame = tk.Frame(left_frame, bg='black')
    camera_control_frame.pack(pady=20)

    # Create a sub-frame to hold icon and text side by side
    camera_toggle_frame = tk.Frame(camera_control_frame, bg='black')
    camera_toggle_frame.pack()

    # Create label for camera feed
    global cam_canvas
    cam_canvas = tk.Label(camera_frame, bg='black')
    cam_canvas.pack(fill="both", expand=True)

    # Create toggle button with custom colors
    global show_camera
    show_camera = tk.BooleanVar(value=True)
    
    global camera_on_img, camera_off_img, camera_icon_label

    camera_on_img = ImageTk.PhotoImage(Image.open("cam_on.png").resize((40, 30)))   # Adjust size as needed
    camera_off_img = ImageTk.PhotoImage(Image.open("cam_off.png").resize((40, 30)))

    # Create label to act as a button with the camera icon
    camera_icon_label = tk.Label(camera_toggle_frame, 
                                image=camera_on_img, 
                                bg='black', 
                                cursor='hand2')
    camera_icon_label.pack(side=tk.LEFT, padx=5)

    # Add text label next to icon
    camera_text_label = tk.Label(camera_toggle_frame,
                                text="Camera Frame",
                                font=("Helvetica", 14),
                                bg='black',
                                fg='white')
    camera_text_label.pack(side=tk.LEFT, padx=5)

    # Bind click event to both icon and text
    camera_icon_label.bind("<Button-1>", lambda event: toggle_camera_display())
    #camera_text_label.bind("<Button-1>", lambda event: toggle_camera_display())

    # Create a frame for the quit button (at the top-right corner)
    quit_frame = tk.Frame(left_frame, bg='black')
    quit_frame.pack(side=tk.LEFT, anchor='ne', padx=20, pady=10)

    # Create quit button with custom styling
    quit_button = tk.Button(quit_frame, 
                          text="Quit (Q)",
                          command=on_closing,
                          font=("Helvetica", 12, "bold"),
                          bg='#FF4444',    # Red background
                          fg='white',       # White text
                          activebackground='#CC0000',  # Darker red when pressed
                          activeforeground='white',
                          width=10,
                          height=1,
                          bd=0,            # No border
                          padx=10,
                          pady=5,
                          cursor='hand2')   # Hand cursor on hover
    quit_button.pack()
    # Add keyboard binding for 'q' key
    root.bind('<q>', lambda event: on_closing())
    root.bind('<Q>', lambda event: on_closing())

def update_gui():
    """
    Updates the GUI with the current tracking status and instructions.
    """
    if CT_side_selected:
        side_label.config(text=f"Patient Side: {CT_side}")
    else:
        side_label.config(text="Patient Side: Please Select Side")
    
    object_label.config(text=f"Selected Part: {selectedObject}")

    if selectedObject == "tibia":
        coronal_label.pack()
        sagittal_label.pack()
        medial_label.pack()
        lateral_label.pack()
        # Update the marker tracking display
        tibia_label.config(text="Tibia Marker", bg="green" if not (corner_point[2] == zeroArr).all() else "red")
        pointer_label.pack()
        rom_label.pack_forget()
        if len(tibia_list) == 0:
            Point_Taken_label.config(text="Take Point: Tibia Knee Center")
            verification_label.pack_forget()
        elif len(tibia_list) == 1:
            Point_Taken_label.config(text="Take Point: Tibia Medial Condyle")
            verification_label.pack_forget()
        elif len(tibia_list) == 2:
            Point_Taken_label.config(text="Take Point: Tibia Lateral Condyle")
            verification_label.pack_forget()
        elif len(tibia_list) == 3:
            if len(ac_Fetcher_List) == 0:
                Point_Taken_label.config(text="Take Point: Medial Malleolus")
                verification_label.pack_forget()
            if len(ac_Fetcher_List) == 1:
                Point_Taken_label.config(text="Take Point: Lateral Malleolus")
                verification_label.pack_forget()
        elif len(tibia_list) == 4:
           Point_Taken_label.config(text="Tibia Points Taken")
           verification_label.pack_forget()  

        if len(tibia_list) < 4:
            pointer_label.config(text="Pointer Marker", bg="green" if not (corner_point[0] == zeroArr).all() else "red")
            #verification_label.config(text="")  # Hide verification marker
        else:
            #pointer_label.config(text="")  # Hide pointer marker
            #pointer_label.config(text="Pointer Marker", bg="green" if not (corner_point[0] == zeroArr).all() else "red")
            verification_label.config(text="Verification Marker", 
                                   bg="green" if not (corner_point[1] == zeroArr).all() else "red")
        if show_reset_text:
            Point_Taken_label.config(text="Resetting...")
            verification_label.pack_forget()
        # Update values
        if len(tibia_list) >= 4:
            coronal_label.config(text=f"Coronal: {varAngle_combined}")
            sagittal_label.config(text=f"Sagittal: {varflex_combined}")
            medial_label.config(text=f"Medial: {Tmc_distance:.0f} mm")
            lateral_label.config(text=f"Lateral: {Tlc_distance:.0f} mm")
            pointer_label.pack_forget()
            verification_label.pack()

    elif selectedObject == "femur":
        coronal_label.pack()
        sagittal_label.pack()
        medial_label.pack()
        lateral_label.pack()
        # Update the marker tracking display
        tibia_label.config(text="Femur Marker", bg="green" if not (corner_point[3] == zeroArr).all() else "red")
        verification_label.pack()
        rom_label.pack_forget()
        if femurFirstFrame is None:
            Point_Taken_label.config(text="Capture First frame of femur")
            verification_label.config(text="stable Marker", 
                                   bg="green" if not (corner_point[1] == zeroArr).all() else "red")
            pointer_label.pack_forget()
            medial_label.config(text="Medial: N/A")
            lateral_label.config(text="Lateral: N/A")
            coronal_label.config(text="Coronal: N/A")
            sagittal_label.config(text="Sagittal: N/A")
        if len(femur_list) == 0 and femurFirstFrame is not None:
            Point_Taken_label.config(text=f"Femur Hip Center Points :{len(hip_center_fetcher_list)+1}")
            verification_label.config(text="stable Marker", 
                                   bg="green" if not (corner_point[1] == zeroArr).all() else "red")
            pointer_label.pack_forget()
            
        elif len(femur_list) == 1:
            Point_Taken_label.config(text="Take Point: Femur Knee Center")
            verification_label.pack_forget()
            pointer_label.pack()
        elif len(femur_list) == 2:
            Point_Taken_label.config(text="Take Point: Femur Distal Medial Condyle")
            verification_label.pack_forget()
            pointer_label.pack()
        elif len(femur_list) == 3:
            Point_Taken_label.config(text="Take Point: Femur Distal Lateral Condyle")
            verification_label.pack_forget()
            pointer_label.pack()
        elif len(femur_list) == 4:
           Point_Taken_label.config(text="Femur Points Taken")
           verification_label.pack_forget()
           pointer_label.pack()
        
        if len(femur_list) < 4:
            pointer_label.config(text="Pointer Marker", bg="green" if not (corner_point[0] == zeroArr).all() else "red")
            #verification_label.config(text="")  # Hide verification marker
        else:
            #pointer_label.config(text="Pointer Marker", bg="green" if not (corner_point[0] == zeroArr).all() else "red")
            verification_label.config(text="Verification Marker", 
                                   bg="green" if not (corner_point[1] == zeroArr).all() else "red")

        if show_reset_text:
            Point_Taken_label.config(text="Resetting...")
            verification_label.pack_forget()
            
        # Update values
        if len(femur_list) >= 4:
            coronal_label.config(text=f"Coronal: {varAngle_combined}")
            sagittal_label.config(text=f"Slope: {varflex_combined}")
            medial_label.config(text=f"Medial: {Dmp_distance:.0f} mm")
            lateral_label.config(text=f"Lateral: {Dlp_distance:.0f} mm")
            pointer_label.pack_forget()
            verification_label.pack()

    if len(tibia_list) >= 4 and len(femur_list) >= 4 and isRomOn:
        # Show ROM label
        rom_label.pack()
        coronal_label.pack_forget()
        sagittal_label.pack_forget()
        medial_label.pack_forget()
        lateral_label.pack_forget()
        # Show marker tracking status for both femur and tibia
        tibia_label.config(text="Tibia Marker", 
                         bg="green" if not (corner_point[2] == zeroArr).all() else "red")
        tibia_label.pack()
        
        pointer_label.pack_forget()  # Hide pointer label during ROM
        
        # Update femur marker status
        verification_label.config(text="Femur Marker",
                                bg="green" if not (corner_point[3] == zeroArr).all() else "red")
        verification_label.pack()
        
        # Hide other labels
        Point_Taken_label.pack_forget()
        
        # Update ROM value if available
        if len(tibia_list) >= 4 and len(femur_list) >= 4:
            rom_label.config(text=f"ROM: {int(rom_angle)}°")
        else:
            rom_label.config(text="ROM: Waiting for markers")
    #root.after(100, update_gui)

def update_camera_feed(frame):
    """Update camera feed in GUI with dynamic sizing"""
    global cam_canvas, show_camera
    
    if show_camera.get():
        try:
            # Get current window dimensions
            frame_width = cam_canvas.winfo_width()
            frame_height = cam_canvas.winfo_height()

            if frame_width > 1 and frame_height > 1:  # Ensure valid dimensions
                # Set minimum size for camera feed
                min_height = int(screen_height * 0.7)  # 70% of screen height
                min_width = int(screen_width * 0.7)   # 70% of screen width

                # Calculate aspect ratio
                frame_aspect = frame.shape[1] / frame.shape[0]
                window_aspect = frame_width / frame_height

                # Determine new dimensions maintaining aspect ratio
                if window_aspect > frame_aspect:
                    # Window is wider than frame
                    new_height = max(frame_height, min_height)
                    new_width = int(new_height * frame_aspect)
                else:
                    # Window is taller than frame
                    new_width = max(frame_width, min_width)
                    new_height = int(new_width / frame_aspect)

                # Resize frame
                display_frame = cv2.resize(frame, (new_width, new_height))
                
                # Convert frame for tkinter display
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(display_frame)
                imgtk = ImageTk.PhotoImage(image=img)
                
                # Update canvas
                cam_canvas.imgtk = imgtk
                cam_canvas.configure(image=imgtk)
        except Exception as e:
            print(f"Error updating camera feed: {str(e)}")
    else:
        cam_canvas.configure(image='')


def on_closing():
    """Handle application closing"""
    try:
        # Show confirmation dialog
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
            print("Closing application...")
            
            # Clean up resources
            root.quit()
            root.destroy()
            cv2.destroyAllWindows()
            pygame.quit()
            
            # Exit the application
            sys.exit()
    except Exception as e:
        print(f"Error during cleanup: {e}")
        sys.exit(1)

show_reset_text = False
reset_start_time = 0
prev_button_b = False
last_button_b_trigger = 0
last_trigger_time = 0
isRomOn = False
femurFirstFrame = None
selectedObject = "tibia" #femur or tibia or none
CT_side = ""  # L for left and R for right
OnlyAtFirstTime = True
CT_side_selected = False 
side_selection_active = True 
last_bumper_time = 0
bumper_cooldown = 1000 

if __name__ == '__main__':
    # Get HoloLens IP before starting the main application
    HOLOLENS_IP = get_ip_address()
    if not HOLOLENS_IP or not is_valid_ip(HOLOLENS_IP):
        print("Invalid or missing IP address. Exiting...")
        sys.exit()

    # Check server connection
    if not check_ip_connection(HOLOLENS_IP, port=9980):
        print(f"Unable to connect to server at {HOLOLENS_IP}:5000. Exiting...")
        sys.exit()
    
    print("Successfully connected to the server!")

    #aruco_ids (pointer, verification, tibia, femur)
    system_config = {
                    "aruco_ids" : [8,5,11,13],
                    "aruco_size" : 50, 
                    "aruco_dictionary" : "DICT_4X4_50"
                    }
    
    if selectedObject == "femur":
        is_taking_Hip_Center = True
  
    # Initialize pygame's joystick module
    pygame.init()
    #pygame.joystick.init()
    # Check if any controller is connected
    if pygame.joystick.get_count() == 0:
        print("No joystick/controller detected.")
        # root = tk.Tk()
        # root.withdraw()  # Hide the root window

        # Show an error dialog box
        messagebox.showerror("Joystick Error", "No joystick/controller detected. Please connect a joystick and restart the application.")
        root.destroy()
        sys.exit()

    joystick = pygame.joystick.Joystick(0)
    joystick.init()  
    print(f"Connected to {joystick.get_name()}")
    
    
    create_gui()
    
    # camera_obj = ComeraCapture_RealSense()
    # update_camera_frames(camera_obj)
    print(system_config)
    
    target_matrix = target_matrix.T
  
    aruco_tibiacorners_ref= None
    aruco_femurcorners_ref= None
    capture_point_flag = False
    #------------
    data_instance = data()
    print(data_instance.StereoObj.New_M1)
    
    exact_matches = None
    marker_size = system_config['aruco_size']

    virtual_sq =  SQ_MARKER_NAVIGATION_SUPPORT_LIB.get_vertual_SQ_coordinates(marker_size)
    print("virtual_sq",virtual_sq)

    pointer_corner_points = np.array(np.load('pv_corner_points_18_4_1_Small_pointer.npy')) # Load the captured points 5X3X4 
    #pointer_corner_points = np.transpose(pointer_corner_points, (0, 2, 1)) # Transpose to 5X4X3
    pivot_point_cam = (SQ_MARKER_NAVIGATION_SUPPORT_LIB.Comput_Aruco_Pivot_Point(pointer_corner_points))
    print("pivot_point_cam : ",pivot_point_cam)

    pivot_def = np.vstack((virtual_sq,pivot_point_cam))
    pivot_point_obj = SQ_MARKER_PIVOT_OBJ(pivot_def)
    T_ref_tib = None
    T_ref_fem = None
    T_ref_vf = None
    tibia_list = []
    femur_list = []
    relative_tibia_points_list = []
    relative_femur_points_list = []
    tibiaCount = None
    femurCount = None
    axial_plane_count = None
    # Initialize the plotter
    plotter = pv.Plotter()
    
    root.protocol("WM_DELETE_WINDOW", root.quit)
    while True:
        # Process pygame events
        
        pygame.event.pump()  # Poll joystick events

        # Check if a controller button is pressed
        # You can replace these checks with the actual logic you're using
        button_a = joystick.get_button(0)  # Example: Button A 
        button_b = joystick.get_button(1)  # Example: Button B 
        button_x = joystick.get_button(2)  # Example: Button X
        button_y = joystick.get_button(3) # Example: Button Y
        left_bumper = joystick.get_button(4)  # Example: Left Bumper
        right_bumper = joystick.get_button(5)  # Example: Right Bumper
        back_button = joystick.get_button(6)  # Example: Back Button
        x_axis, y_axis = joystick.get_hat(0)

        # Encode strings to fixed-length byte strings
        selected_object_encoded = selectedObject.encode('utf-8').ljust(16, b'\x00')  # 16 bytes
        ct_side_encoded = CT_side.encode('utf-8').ljust(8, b'\x00')  # 8 bytes

        ret, left, right ,is_marker_detected = data_instance.obj2.get_IR_FRAME_SET()
        if show_camera.get():
            update_camera_feed(left)
        corner_point, center_points ,is_marker_detected = data_instance.send_data()
        #Draw borders for the markers
        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(left, dictionary, [8,5,11,13])
        # Increase thickness manually using cv2.polylines
        cv2.aruco.drawDetectedMarkers(left, corners, ids, borderColor=(255, 255, 255)) 
        if corners:
            for corner in corners:
                pts = corner[0].astype(int)  # Convert to int
                cv2.polylines(left, [pts], isClosed=True, color=(255, 255, 255), thickness=4)
        #draw3dPoseMarkers(data_instance.obj2.camera_matrix, data_instance.obj2.dist_coeffs, left, corners, ids, marker_size)
        
        exact_matches = [np.array_equal(arr, zeroArr) for arr in corner_point]
        
        #sending verification paddle values to unity constantly
        
        if selectedObject == "tibia":
            aruco_tibiacorners_ref = corner_point[2]  # Capture reference marker points 
            T_ref_tib = SQ_MARKER_NAVIGATION_SUPPORT_LIB.GET_ARUCO_VREF_TRANSFORM(aruco_tibiacorners_ref, marker_size)
        elif selectedObject == "femur":
            aruco_femurcorners_ref = corner_point[3]  # Capture reference marker points 
            T_ref_fem = SQ_MARKER_NAVIGATION_SUPPORT_LIB.GET_ARUCO_VREF_TRANSFORM(aruco_femurcorners_ref, marker_size)
             
        #Corner points of verification
        if not (corner_point[1] == zeroArr).all():
            if selectedObject == "tibia":
                vp_corners = getRelativeToMarker(corner_point[1],corner_point[2])
                #vp_corners = corner_point[1]
                convertedToTuple = tuple(map(tuple,vp_corners))
                center_points[1] = np.mean(vp_corners, axis=0)
                vp_normal = Planes_genrate_LIB.calculate_plane_normal(vp_corners)
                
            elif selectedObject == "femur":
                vp_corners = getRelativeToMarker(corner_point[1],corner_point[3])
                convertedToTuple = tuple(map(tuple,vp_corners))
                center_points[1] = np.mean(vp_corners, axis=0)
                vp_normal = Planes_genrate_LIB.calculate_plane_normal(vp_corners)
                
                
        if selectedObject == "tibia":
            if not (corner_point[0] == zeroArr).all() and T_ref_tib is not None:
                tibia_pointtaker()
            tibia_verifycuts()
            tibia_point_referencing()    
        elif selectedObject == "femur":
            #if not (corner_point[1] == zeroArr).all() or not (corner_point[3] == zeroArr).all() and T_ref_vf is not None and len(femur_list) == 0:
            if not (corner_point[0] == zeroArr).all() or not (corner_point[3] == zeroArr).all() and T_ref_fem is not None:
                femur_pointtaker()
            femur_verifycuts()
            femur_point_referencing()

        if button_x and len(femur_list) !=4  and len(tibia_list) == 4:
            # Send message to Unity
            message1 = "femur"
            message1_encoded = message1.encode('utf-8').ljust(16, b'\x00')
            message = "signal Off"
            message_encoded = message.encode('utf-8').ljust(32, b'\x00')
            # Pack the buffer with the message  

            buffer = struct.pack('32s16s', message_encoded,message1_encoded)
            # Send the buffer to Unity
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                #s.connect((getTheIP(), PORT))
                s.connect((HOLOLENS_IP, PORT))
                s.sendall(buffer)
                print("signal Off message sent to Unity.")

        current_time = pygame.time.get_ticks()
        if not CT_side_selected:
            pygame.event.pump()
                
            # Handle bumper input with cooldown
            if (left_bumper or right_bumper) and (current_time - last_bumper_time >= bumper_cooldown):
                if left_bumper:
                    CT_side = "L"
                    side_selection_active = False
                    print("Left side selected")
                elif right_bumper:
                    CT_side = "R"
                    #CT_side_selected = True
                    side_selection_active = False
                    print("Right side selected")
                CT_side_selected = True       
                last_bumper_time = current_time
        

        #Controls 
        key = cv2.waitKey(1) & 0xFF
        
        if button_x:
            print("Button A pressed")
            # Trigger femur points action or any other function here
            isRomOn = False
            selectedObject = "femur"
        
        elif button_y:
            print("Button B pressed")
            # Trigger tibia points action or any other function here
            isRomOn = False
            selectedObject = "tibia"
        
        pygame.event.pump()  # Poll joystick events
        current_time = pygame.time.get_ticks()

        # Get current button state for 'b' button (assume it's button index 1)
        button_b_current = joystick.get_button(1)

        # Detect rising edge
        button_b_pressed = button_b_current and not prev_button_b
        prev_button_b = button_b_current  # Update previous state

        # Define your cooldown time (in milliseconds)
        cooldown_duration = 1000

        # Check trigger condition with timer
        key_pressed = (key == ord('m'))
        button_allowed = button_b_pressed and (current_time - last_button_b_trigger >= cooldown_duration)

        if (key_pressed or button_allowed):
            if current_time - last_button_b_trigger >= cooldown_duration:
                # Toggle isRomOn with 1-second cooldown
                isRomOn = not isRomOn
                selectedObject = "none" if isRomOn else selectedObject

                # Update last trigger time if button pressed
                last_button_b_trigger = current_time
                                  
        if key == ord('/') or button_a and selectedObject == "femur" and OnlyAtFirstTime and not (corner_point[1] == zeroArr).all() and CT_side_selected: 
            femur_refernce = getRelativeToMarker(corner_point[3],corner_point[1])
            femurFirstFrame = femur_refernce
            print("Femur First Frame Captured : ",  femurFirstFrame)
            OnlyAtFirstTime = False
        
        if key == ord('r') or x_axis == 1:
            show_reset_text = True
            reset_start_time = time.time()  # Record reset time
            # Send reset message to Unity
            reset_message = "Reset Points"
            reset_message_encoded = reset_message.encode('utf-8').ljust(32, b'\x00')  # Encode and pad the message

            # Pack the buffer with the reset message
            buffer = struct.pack('32s', reset_message_encoded)

            # Send the buffer to Unity
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                #s.connect((getTheIP(), PORT))
                s.connect((HOLOLENS_IP, PORT))
                s.sendall(buffer)
                print("Reset message sent to Unity.")
                
            if selectedObject == "femur": 
                femurFirstFrame = None
                femur_list = []
                relative_femur_points_list = []
                hip_center_fetcher_list = []
                hip_center_fetcher_print_list = []
                femurCount = None
                is_taking_Hip_Center = True
                OnlyAtFirstTime = True
                print("Reset femur")
            if selectedObject == "tibia":
                tibia_list = []
                relative_tibia_points_list = []  
                tibiaCount = None
                print("Reset Tibia")

        if show_reset_text:
            elapsed_time = time.time() - reset_start_time
            if selectedObject == "femur":
                if elapsed_time < 2:
                    True
                else:
                    show_reset_text = False  # Stop showing after 2 seconds
            if selectedObject == "tibia":
                if elapsed_time < 2:
                    True
                else:
                    show_reset_text = False  # Stop showing after 2 seconds
                    
        trigger_pressed = key == ord('m') or y_axis == 1
        #for taking back a single particular point
        if trigger_pressed and (current_time - last_trigger_time >= cooldown_duration):
            last_trigger_time = current_time  # Update the cooldown timestamp
            #femur point removal
            if selectedObject == "femur" and len(femur_list) == 1:
                femurFirstFrame = None
                removedFirstValue = False
                hip_center_fetcher_list = []
                hip_center_fetcher_print_list = [] 
                is_taking_Hip_Center = True
                femur_list = []
                relative_femur_points_list = []
                OnlyAtFirstTime = True
            elif selectedObject == "femur" and len(femur_list) > 0 and len(femur_list) < 4 and is_taking_Hip_Center == False:
                remove_point = femur_list.pop()
                relative_femur_points_list.pop()     
            elif selectedObject == "femur" and len(femur_list) == 0 and len(hip_center_fetcher_list) > 0:
                hip_center_fetcher_list.pop()
                hip_center_fetcher_print_list.pop()

            #tibia point removal    
            if selectedObject == "tibia" and len(tibia_list) > 0 and len(tibia_list) < 4 and len(ac_Fetcher_List) == 0:
                remove_point = tibia_list.pop()
                relative_tibia_points_list.pop()
                print("Removed point:", remove_point)
            elif len(ac_Fetcher_List) > 0 :
                remove_point=ac_Fetcher_List.pop()
                print("Removed point:", remove_point)
            
        
        if isRomOn and corner_point[2] is not None and corner_point[3] is not None:
            global rom_angle
            try: 
                New_femur_hip = virtual_to_camera(relative_femur_points_list[0],corner_point[3])
                New_femur_knee = virtual_to_camera(relative_femur_points_list[1],corner_point[3])
                New_tibia_knee = virtual_to_camera(relative_tibia_points_list[0],corner_point[2])
                New_tibia_ankle = virtual_to_camera(relative_tibia_points_list[3],corner_point[2])
                
                New_femur_hip = getRelativeToPoint(New_femur_hip,corner_point[3])
                New_femur_knee = getRelativeToPoint(New_femur_knee,corner_point[3])
                New_tibia_knee = getRelativeToPoint(New_tibia_knee,corner_point[3])
                New_tibia_ankle = getRelativeToPoint(New_tibia_ankle,corner_point[3])

                rom_angle = ROM(New_femur_hip, New_femur_knee, 
                            New_tibia_knee, New_tibia_ankle)
                
                coronal_rom_angle = Angle(New_femur_hip, New_femur_knee,
                            New_tibia_knee, New_tibia_ankle,coronal_Dir,CT_side)[0]

                #Send ROM angle via buffer
                rom_angle_message = f"ROM: {int(rom_angle)}"
                rom_angle_message_encoded = rom_angle_message.encode('utf-8').ljust(32, b'\x00')  # Encode and pad the message

                coronal_rom_angel_message = f"angle: {(coronal_rom_angle)}"
                coronal_rom_angle_message_encoded = coronal_rom_angel_message.encode('utf-8').ljust(32, b'\x00')  # Encode and pad the message
                
                # Pack the buffer with the ROM angle
                buffer = struct.pack('32s32s', rom_angle_message_encoded,coronal_rom_angle_message_encoded)
               

                # Send the buffer to Unity
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect((HOLOLENS_IP, PORT))
                    s.sendall(buffer)
                    print(f"ROM angle sent: {rom_angle_message}")
                    print(f"Coronal angle sent: {coronal_rom_angel_message}")

            except Exception as e:
                print(f"ROM Calculation Error: {str(e)}")
      
        # Start the Tkinter main loop
        update_gui()
        
        root.update()
        #cv2.imshow('left',cv2.resize(left,(1280,720)))
        
        #root.update()
        #close application when 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        
        # Check for 'q' press or back button on controller
        if key == ord('q') or back_button:
            print("Closing application...")
            on_closing()
            break
     
        time.sleep(0.05)
    # Cleanup
    # cv2.destroyAllWindows()
    # root.destroy()      
