import numpy as np
import json
import math
import cv2
import os
import itertools
import time
from threading import Thread
import socket
from enum import Enum
# import rigid_body_transform_info # Integrated into TransformationUtils
# import Trans3D # Integrated into TransformationUtils
import configparser
from main1 import data # External dependency, assumed to exist
# from CAMERA_CAPTURE_MODULE_MAIN import ComeraCapture_RealSense # External dependency, assumed to exist
from collections import deque
import SQ_MARKER_NAVIGATION_SUPPORT_LIB
from SQ_MARKER_NAVIGATION_SUPPORT_LIB import SQ_MARKER_PIVOT_OBJ
import REG_AND_TRANS_SUPPORT_LIB # Still used for some plane calculations
import Planes_genrate_LIB # Still used for plane calculations
import pyvista as pv # Not directly used in the main loop for display, but imported
from sympy import Plane, Point, Point3D # Used for plane definitions
import pygame
import struct
import tkinter as tk
from tkinter import messagebox
import tkinter.font as tkFont
from PIL import Image, ImageTk
import threading
import sys

# --- Utility Classes ---

class TransformationUtils:
    """
    Utility class for rigid body transformations and geometric calculations.
    All methods are static as they don't depend on instance state.
    """
    @staticmethod
    def cross_times_matrix(V: np.ndarray) -> np.ndarray:
        """
        Computes the cross-product matrix for a given vector V.
        Used in rigid body transformation estimation.
        Args:
            V (np.ndarray): A 3xN numpy array representing vectors.
        Returns:
            np.ndarray: A 3x3xN numpy array where each 3x3 slice is the
                        cross-product matrix for the corresponding column vector in V.
        """
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

    @staticmethod
    def quat_to_rot(Q: np.ndarray) -> np.ndarray:
        """
        Converts a quaternion to a 3x3 rotation matrix.
        Args:
            Q (np.ndarray): A 1x4 numpy array representing the quaternion [q0, q1, q2, q3].
        Returns:
            np.ndarray: A 3x3 rotation matrix.
        """
        q0, q1, q2, q3 = Q[0], Q[1], Q[2], Q[3]
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

    @staticmethod
    def compute_pivot_point(P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
        """
        Computes a pivot point from two sets of corresponding points using least squares.
        Args:
            P1 (np.ndarray): Nx3 array of points.
            P2 (np.ndarray): Nx3 array of points.
        Returns:
            np.ndarray: The computed pivot point (1x3 array).
        """
        B = np.sum((np.square(P2) - np.square(P1)), axis=1)/2
        ans = np.linalg.lstsq(P2-P1, B, rcond=None)
        return ans[0]

    @staticmethod
    def estimate_rigid_transform(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Estimates the rigid body transformation (rotation and translation) between two
        corresponding 3D point sets using the method of Horn.
        Args:
            x (np.ndarray): 3xN numpy array of source points.
            y (np.ndarray): 3xN numpy array of target points.
        Returns:
            tuple[np.ndarray, float]: A tuple containing:
                - T (np.ndarray): The 4x4 homogeneous transformation matrix.
                - Eps (float): The error (eigenvalue related to the smallest singular value).
        Raises:
            Exception: If input point clouds have incorrect dimensions or size.
        """
        if (x.shape[0] != 3) or (y.shape[0] != 3):
            raise Exception("Input point clouds must be a 3xN matrix.")
        if x.shape[1] != y.shape[1]:
            raise Exception("Input point clouds must be of the same size")
        if (x.shape[1] < 3) or (y.shape[1] < 3):
            raise Exception("At least 3 point matches are needed")

        point_count = x.shape[1]

        x_centroid = (np.sum(x, axis=1)/point_count).reshape(3, 1)
        y_centroid = (np.sum(y, axis=1)/point_count).reshape(3, 1)

        x_centrized = x - x_centroid
        y_centrized = y - y_centroid

        R12 = y_centrized.T - x_centrized.T
        R21 = x_centrized - y_centrized
        R22_1 = y_centrized + x_centrized
        R22 = TransformationUtils.cross_times_matrix(R22_1)

        B = np.zeros(shape=(4, 4))
        A = np.zeros(shape=(4, 4, point_count))

        for i in range(point_count):
            A[:, :, i][0][0] = 0
            A[:, :, i][0, 1:] = R12[i, :]
            A[:, :, i][1:, 0] = R21[:, i]
            A[:, :, i][1:, 1:] = R22[:, :, i]
            B = B + np.dot((A[:, :, i]).T, (A[:, :, i]))

        [U, s, v] = np.linalg.svd(B, full_matrices=True)
        S = np.diag(s)
        V = v.T

        quat = V[:, 3]
        rot = TransformationUtils.quat_to_rot(quat)

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

    @staticmethod
    def apply_affine_transform(points: np.ndarray, T_mat: np.ndarray) -> np.ndarray:
        """
        Applies a 4x4 affine transformation matrix to a set of 3D points.
        Args:
            points (np.ndarray): A 3xN numpy array of points.
            T_mat (np.ndarray): A 4x4 affine transformation matrix.
        Returns:
            np.ndarray: A 3xN numpy array of transformed points.
        """
        ndim = points.shape[0]
        num_point = points.shape[1]
        tmp = np.concatenate((points.T, np.ones([num_point, 1])), axis=1)
        tmp = np.dot(tmp, T_mat.T)
        transformed_points = tmp[:, :ndim]
        transformed_points = transformed_points.T
        return transformed_points

    @staticmethod
    def get_pivot_point(PP: np.ndarray) -> dict:
        """
        Estimates a pivot point from multiple sets of point cloud data.
        Args:
            PP (np.ndarray): A Nx3xM array of point clouds, where N is the number of examples,
                             3 is the dimension, and M is the number of points per example.
        Returns:
            dict: A dictionary containing "pv_point" (the median pivot point) and "pv_tool" (the first point cloud).
        """
        pv_tool = np.squeeze(PP[0, :, :]).T
        p_pairs = np.array(list(itertools.combinations(np.arange(0, PP.shape[0]), 2)))

        num_example = 10
        pivot_points = None

        for k in range(20):
            rp = np.arange(np.size(p_pairs, axis=0))
            rp = np.random.permutation(rp)
            rp = rp[:num_example]

            rp_a = p_pairs[rp, 0]
            rp_b = p_pairs[rp, 1]

            P1 = np.array(np.squeeze(PP[rp_a[0], :, :]).T)
            P2 = np.array(np.squeeze(PP[rp_b[0], :, :]).T)

            for i in range(1, num_example):
                P1 = np.concatenate((P1, np.squeeze(PP[rp_a[i], :, :]).T), axis=0)
                P2 = np.concatenate((P2, np.squeeze(PP[rp_b[i], :, :]).T), axis=0)

            X = TransformationUtils.compute_pivot_point(P1, P2)
            X = X.reshape(X.shape[0], 1)

            R, eps = TransformationUtils.estimate_rigid_transform(pv_tool.T, np.squeeze(PP[0, :, :]))
            tmp_pivot_points = np.array(TransformationUtils.apply_affine_transform(X, R)).T

            for j in range(1, PP.shape[0]):
                R, eps = TransformationUtils.estimate_rigid_transform(pv_tool.T, np.squeeze(PP[j, :, :]))
                tmp_pivot_points = np.concatenate((tmp_pivot_points, np.array(TransformationUtils.apply_affine_transform(X, R)).T), axis=0)

            if k == 0:
                pivot_points = np.mean(tmp_pivot_points, axis=0).reshape(3, 1).T
            else:
                pivot_points = np.concatenate((pivot_points, np.mean(tmp_pivot_points, axis=0).reshape(3, 1).T))

        pv = {}
        pv["pv_point"] = np.median(pivot_points, axis=0)
        pv["pv_tool"] = pv_tool
        return pv

    @staticmethod
    def get_average_of_four_corners(fourCorners: np.ndarray) -> tuple:
        """
        Calculates the average (center) of four 3D corner points.
        Args:
            fourCorners (np.ndarray): A 3x4 numpy array representing the x,y,z coordinates
                                      of four corners.
        Returns:
            tuple: A tuple (x, y, z) representing the center point.
        """
        result = np.array([0.0, 0.0, 0.0])
        for i in range(3):
            result[i] = np.average(fourCorners[i])
        return tuple(result)

    @staticmethod
    def get_relative_to_marker(point_matrix: np.ndarray, ref_marker: np.ndarray, target_matrix: np.ndarray) -> np.ndarray:
        """
        Transforms a point matrix relative to a reference marker using a target matrix.
        Args:
            point_matrix (np.ndarray): 4x3 array of points to transform.
            ref_marker (np.ndarray): 4x3 array of reference marker points.
            target_matrix (np.ndarray): 4x3 array defining the target coordinate system.
        Returns:
            np.ndarray: Transformed 4x3 matrix of points.
        """
        point_matrix_t = point_matrix.T
        ref_marker_t = ref_marker.T
        
        # compute_rigid_body_transform expects 3xN arrays
        T_mat = TransformationUtils.compute_rigid_body_transform(ref_marker_t, target_matrix.T)
        T_points = TransformationUtils.apply_affine_transform(point_matrix_t, T_mat)
        return T_points.T

    @staticmethod
    def find_ankle_center(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Calculates the ankle center point, 44% away from point A towards point B.
        Args:
            A (np.ndarray): 1x3 numpy array representing the first point.
            B (np.ndarray): 1x3 numpy array representing the second point.
        Returns:
            np.ndarray: 1x3 numpy array representing the calculated ankle center.
        """
        alpha = 0.44
        M = (1 - alpha) * A + alpha * B
        return M

    @staticmethod
    def get_relative_position(parent_matrix: np.ndarray, child_point: np.ndarray) -> np.ndarray:
        """
        Computes the relative position of the child point with respect to the parent marker.
        Args:
            parent_matrix (np.ndarray): 4x3 matrix of the parent marker's corners.
            child_point (np.ndarray): 1x3 array representing the child's position.
        Returns:
            np.ndarray: 1x3 relative position of the child.
        """
        parent_matrix_t = parent_matrix.T
        child_point_reshaped = child_point.reshape(3, 1)

        T_mat = TransformationUtils.compute_rigid_body_transform(parent_matrix_t, parent_matrix_t)
        relative_position_hom = np.linalg.inv(T_mat) @ np.vstack((child_point_reshaped, [[1]]))
        return relative_position_hom[:3].flatten()

    @staticmethod
    def update_child_position(parent_matrix: np.ndarray, relative_position: np.ndarray) -> np.ndarray:
        """
        Updates the child’s absolute position based on the parent’s new transformation
        and the stored relative position.
        Args:
            parent_matrix (np.ndarray): Updated 4x3 matrix of the parent marker.
            relative_position (np.ndarray): Stored relative position (1x3).
        Returns:
            np.ndarray: Updated absolute position of the child (1x3).
        """
        parent_matrix_t = parent_matrix.T
        T_mat = TransformationUtils.compute_rigid_body_transform(parent_matrix_t, parent_matrix_t)
        updated_position_hom = T_mat @ np.vstack((relative_position.reshape(3, 1), [[1]]))
        return updated_position_hom[:3].flatten()

    @staticmethod
    def compute_rigid_body_transform(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
        """
        Computes the 4x4 rigid transformation matrix between two sets of 3D points.
        Args:
            source_points (np.ndarray): 3xN array of source points.
            target_points (np.ndarray): 3xN array of target points.
        Returns:
            np.ndarray: 4x4 homogeneous transformation matrix.
        """
        assert source_points.shape == target_points.shape, "Source and target must have the same shape"

        src_centroid = np.mean(source_points, axis=1, keepdims=True)
        tgt_centroid = np.mean(target_points, axis=1, keepdims=True)

        x_centered = source_points - src_centroid
        y_centered = target_points - tgt_centroid

        H = y_centered @ x_centered.T
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = tgt_centroid - R @ src_centroid

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        return T

    @staticmethod
    def calculate_angle(A: np.ndarray, B: np.ndarray) -> float:
        """
        Calculate the angle between two 3D vectors in degrees.
        Args:
            A (np.ndarray): First 3D vector.
            B (np.ndarray): Second 3D vector.
        Returns:
            float: Angle in degrees.
        """
        dot_product = np.dot(A, B)
        norm_A = np.linalg.norm(A)
        norm_B = np.linalg.norm(B)
        cos_theta = dot_product / (norm_A * norm_B)
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        return np.degrees(theta)

    @staticmethod
    def calculate_signed_angle(A: np.ndarray, B: np.ndarray, normal_vector: np.ndarray) -> float:
        """
        Calculate signed angle between two 3D vectors in degrees.
        The sign is determined by the direction of the cross product relative to the normal_vector.
        Args:
            A (np.ndarray): First 3D vector.
            B (np.ndarray): Second 3D vector.
            normal_vector (np.ndarray): Normal vector defining the plane for the signed angle.
        Returns:
            float: Signed angle in degrees.
        """
        dot = np.dot(A, B)
        norm_A = np.linalg.norm(A)
        norm_B = np.linalg.norm(B)
        cos_theta = dot / (norm_A * norm_B)
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        cross = np.cross(A, B)
        direction = np.sign(np.dot(normal_vector, cross))

        return np.degrees(theta) * direction

    @staticmethod
    def calculate_distance(marker1: np.ndarray, marker2: np.ndarray) -> float:
        """
        Calculate the Euclidean distance between two markers in 3D space.
        Args:
            marker1 (np.ndarray): 3D coordinates of the first marker.
            marker2 (np.ndarray): 3D coordinates of the second marker.
        Returns:
            float: The Euclidean distance.
        """
        marker1 = np.array(marker1)
        marker2 = np.array(marker2)
        distance = np.linalg.norm(marker1 - marker2)
        return distance

    @staticmethod
    def virtual_to_camera(point_virtual: np.ndarray, camera_marker_corners: np.ndarray) -> np.ndarray:
        """
        Converts a point from a virtual coordinate system to camera coordinates.
        Args:
            point_virtual (np.ndarray): 1x3 point in virtual coordinate system.
            camera_marker_corners (np.ndarray): 4x3 detected marker corners in camera coordinates.
        Returns:
            np.ndarray: 1x3 point in camera coordinates.
        """
        virtual_marker_local = np.array([
            [-30.0, 30.0, 0],   # Top-left
            [30.0, 30.0, 0],    # Top-right
            [30.0, -30.0, 0],   # Bottom-right
            [-30.0, -30.0, 0]   # Bottom-left
        ])

        T = TransformationUtils.compute_rigid_body_transform(
            source_points=virtual_marker_local.T,
            target_points=camera_marker_corners.T
        )

        homogeneous_point = np.vstack([point_virtual[np.newaxis, :].T, np.ones((1, 1))])
        transformed_homogeneous = T @ homogeneous_point
        return transformed_homogeneous[:3, :].flatten()

    @staticmethod
    def get_relative_to_point(point_matrix: np.ndarray, ref_marker: np.ndarray) -> np.ndarray:
        """
        Transforms a single point relative to a reference marker.
        Args:
            point_matrix (np.ndarray): 1x3 point to transform.
            ref_marker (np.ndarray): 4x3 array of reference marker points.
        Returns:
            np.ndarray: Transformed 1x3 point.
        """
        # Define a target matrix for the reference marker's local frame (e.g., identity or known virtual points)
        # For simplicity, let's assume `ref_marker` itself defines the target frame.
        # This function's original implementation was `getRelativeToPoint` which used `corner_point[3]` as target.
        # This needs careful re-evaluation based on the actual coordinate system desired.
        # For now, let's assume `ref_marker` is the new origin/orientation.
        
        # This part of the original code was confusing. `compute_rigid_body_TANSFORM` and `apply_affine_TANSFORM_single_point`
        # were specific to single point transformations. Let's merge that logic here.
        
        # We need a source frame (e.g., the camera frame where `point_matrix` is)
        # and a target frame (defined by `ref_marker`).
        # Let's use a simplified approach assuming `ref_marker` gives us the transformation.

        # A common way to get a point in a marker's local frame:
        # 1. Get the transformation from marker's local frame to camera frame (T_marker_cam)
        # 2. Invert it to get T_cam_marker
        # 3. Apply T_cam_marker to the point in camera frame.

        # To avoid re-implementing `compute_rigid_body_transform` for this specific case,
        # and given the original code's structure, let's stick to the spirit of its `_TANSFORM` functions.
        
        # The `compute_rigid_body_TANSFORM` was a duplicate of `compute_rigid_body_transform`
        # but with different naming and potentially usage.
        # The `apply_affine_TANSFORM_single_point` was also a specific version.
        
        # Let's use the general `compute_rigid_body_transform` and `apply_affine_transform`
        # by ensuring inputs are correctly shaped.

        # Define a virtual representation of the marker in its own local coordinate system
        # (assuming a standard square marker centered at origin on XY plane)
        marker_local_coords = np.array([
            [-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0] # Normalized example
        ]) * 30 # Scale to match original marker size if needed, or use actual marker size

        # Compute transformation from marker's local frame to the camera frame (where ref_marker is)
        T_marker_to_camera = TransformationUtils.compute_rigid_body_transform(
            marker_local_coords.T, ref_marker.T # Source is marker_local, Target is ref_marker in camera
        )
        
        # To get a point from camera frame (where `point_matrix` is) to marker's local frame,
        # we need the inverse transformation.
        T_camera_to_marker = np.linalg.inv(T_marker_to_camera)

        # Apply this inverse transformation to the single point
        point_transformed = TransformationUtils.apply_affine_transform(
            point_matrix.reshape(3, 1), # Reshape 1x3 to 3x1 for apply_affine_transform
            T_camera_to_marker
        )
        return point_transformed.flatten() # Return as 1x3

    @staticmethod
    def project_point_onto_plane(point: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
        """
        Projects a 3D point onto a plane.
        Args:
            point (np.ndarray): The 3D point to project.
            plane_point (np.ndarray): A point on the plane.
            plane_normal (np.ndarray): The normal vector of the plane.
        Returns:
            np.ndarray: The projected 3D point.
        """
        vector_to_point = point - plane_point
        distance = np.dot(vector_to_point, plane_normal)
        projected_point = point - distance * plane_normal
        return projected_point

    @staticmethod
    def signed_angle_plane(a: np.ndarray, b: np.ndarray, axis: np.ndarray) -> float:
        """
        Calculates the signed angle between two vectors projected onto a plane defined by an axis.
        Args:
            a (np.ndarray): First 3D vector.
            b (np.ndarray): Second 3D vector.
            axis (np.ndarray): Normal vector of the plane onto which vectors are projected.
        Returns:
            float: Signed angle in degrees.
        """
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

    @staticmethod
    def Angle(hip: np.ndarray, knee: np.ndarray, tibia_knee: np.ndarray, ankle: np.ndarray, coronal_normal: np.ndarray, side: str) -> tuple[str, float]:
        """
        Calculates the Varus/Valgus angle based on mechanical axes and a coronal plane normal.
        Args:
            hip (np.ndarray): Hip center point.
            knee (np.ndarray): Knee center point.
            tibia_knee (np.ndarray): Tibia knee point.
            ankle (np.ndarray): Ankle center point.
            coronal_normal (np.ndarray): Normal vector of the coronal plane.
            side (str): "L" for Left, "R" for Right.
        Returns:
            tuple[str, float]: A formatted string indicating angle and type (Varus/Valgus),
                               and the raw signed angle.
        """
        femur_vector = knee - hip
        tibia_vector = tibia_knee - ankle

        angle_var_val = TransformationUtils.signed_angle_plane(femur_vector[0], tibia_vector[0], coronal_normal)
        displayed_value = 180 - abs(angle_var_val)
        displayed_value = abs(displayed_value)

        angle_str = f"{displayed_value:.1f}°"

        if side == "L":
            if angle_var_val > 0: # Positive angle for left means valgus
                angle_str = f"{displayed_value:.0f} Valgus"
            else: # Negative angle for left means varus
                angle_str = f"{displayed_value:.0f} Varus"
        else:  # Right side
            if angle_var_val > 0: # Positive angle for right means varus
                angle_str = f"{displayed_value:.0f} Varus"
            else: # Negative angle for right means valgus
                angle_str = f"{displayed_value:.0f} Valgus"

        return angle_str, angle_var_val


# --- Core Application Classes ---

class MarkerTracker:
    """
    Manages Aruco marker detection and provides marker corner and center points.
    Assumes `data` object from `main1` and `ComeraCapture_RealSense` are accessible.
    """
    
    def __init__(self, system_config: dict, camera_capture_obj):
        self.system_config = system_config
        self.camera_capture_obj = camera_capture_obj
        self.aruco_dictionary = cv2.aruco.Dictionary_get(getattr(cv2.aruco, system_config['aruco_dictionary']))
        self.marker_length = system_config['aruco_size']
        self.zero_array = np.zeros((4,3))

    def get_marker_data(self) -> tuple[np.ndarray, list, list, list, bool]:
        """
        Captures camera frames, detects Aruco markers, and returns their data.
        Args:
            None
        Returns:
            tuple: (left_frame, corner_point, center_points, exact_matches, is_marker_detected)
                - left_frame (np.ndarray): The processed camera frame.
                - corner_point (list): List of 4x3 arrays for each detected marker's corners.
                - center_points (list): List of 1x3 arrays for each detected marker's center.
                - exact_matches (list): Boolean list indicating if a marker was detected (False if zero_array).
                - is_marker_detected (bool): Overall flag if any marker was detected.
        """
        ret, left_frame, right_frame, is_marker_detected_cam = data_instance.obj2.get_IR_FRAME_SET()
        corner_point, center_points, is_marker_detected_data = data_instance.send_data()

        # Draw borders for the markers
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(left_frame, self.aruco_dictionary, self.system_config['aruco_ids'])
        
        # Ensure 'ids' is not None before iterating
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(left_frame, corners, ids, borderColor=(255, 255, 255))
            for corner in corners:
                pts = corner[0].astype(int)
                cv2.polylines(left_frame, [pts], isClosed=True, color=(255, 255, 255), thickness=4)
            # Optionally draw 3D pose axes
            # self.draw_pose_markers(left_frame, corners, ids)

        exact_matches = [np.array_equal(arr, self.zero_array) for arr in corner_point]
        return left_frame, corner_point, center_points, exact_matches, (is_marker_detected_cam or is_marker_detected_data)

    def draw_pose_markers(self, img: np.ndarray, corners: list, ids: np.ndarray):
        """
        Draws 3D axes on detected Aruco markers for visualization.
        Args:
            img (np.ndarray): The image frame to draw on.
            corners (list): List of detected marker corners.
            ids (np.ndarray): Array of detected marker IDs.
        """
        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.camera_capture_obj.camera_matrix, self.camera_capture_obj.dist_coeffs)
            for i in range(len(ids)):
                cv2.aruco.drawAxis(img, self.camera_capture_obj.camera_matrix, self.camera_capture_obj.dist_coeffs, rvecs[i], tvecs[i], 0.05)


class PointManager:
    """
    Manages the collection and referencing of anatomical points for tibia and femur.
    Relies on TransformationUtils and SQ_MARKER_NAVIGATION_SUPPORT_LIB.
    """
    def __init__(self, system_config: dict, pivot_point_obj: SQ_MARKER_PIVOT_OBJ):
        self.tibia_list = []
        self.femur_list = []
        self.relative_tibia_points_list = []
        self.relative_femur_points_list = []
        self.ac_fetcher_list = [] # For Ankle Center calculation
        self.hip_center_fetcher_list = [] # For Hip Center calculation
        self.hip_center_fetcher_print_list = [] # For debugging/display of HC points
        self.is_taking_hip_center = True # Flag for HC collection phase
        self.femur_first_frame = None # Reference frame for femur point collection
        self.pivot_point_obj = pivot_point_obj
        self.zero_array = np.zeros((4,3))
        self.target_matrix = np.array([
            [-30.0,30.0,0],
            [30.0,30.0,0],
            [30.0,-30.0,0],
            [-30.0,-30.0,0]
        ]) # Note: Original code used .T in `get_relative_to_marker`

        # Joystick state for point taking (moved from global)
        self.prev_button_a = False
        self.last_button_a_trigger = 0

    def tibia_point_taker(self, corner_point: list, joystick_obj, ct_side_selected: bool, T_ref_tib: np.ndarray):
        """
        Collects anatomical points for the tibia based on joystick input.
        Args:
            corner_point (list): Current list of detected marker corners.
            joystick_obj: Pygame joystick object.
            ct_side_selected (bool): True if patient side is selected.
            T_ref_tib (np.ndarray): Transformation matrix for tibia reference.
        """
        pointer_corners = corner_point[0]
        if (pointer_corners == self.zero_array).all() or (corner_point[2] == self.zero_array).all():
            return # Do not proceed if pointer or tibia marker is not visible

        try:
            pivot_cam = self.pivot_point_obj.get_pivot_point_CamCoordSys(pointer_corners)
            pivot_ref = SQ_MARKER_NAVIGATION_SUPPORT_LIB.apply_tranform_to_points(pivot_cam, T_ref_tib)
            pivot_ref = pivot_ref[4,:] # Assuming this is the correct pivot point from the transformed array

            pygame.event.pump()
            current_time = pygame.time.get_ticks()
            button_a_current = joystick_obj.get_button(0) # Button A for point taking
            button_a_pressed = button_a_current and not self.prev_button_a
            self.prev_button_a = button_a_current

            button_allowed = button_a_pressed and (current_time - self.last_button_a_trigger >= 1000) # 1-second cooldown

            if button_allowed and len(self.tibia_list) <= 6 and ct_side_selected:
                self.last_button_a_trigger = current_time # Update cooldown

                if len(self.tibia_list) == 0:
                    print("KC (Knee Center) point taken.")
                    self.tibia_list.append(pivot_ref)
                    relative_pos = TransformationUtils.get_relative_position(corner_point[2], pivot_ref)
                    self.relative_tibia_points_list.append(relative_pos)
                elif len(self.tibia_list) == 1:
                    print("TMC (Tibia Medial Condyle) point taken.")
                    self.tibia_list.append(pivot_ref)
                    relative_pos = TransformationUtils.get_relative_position(corner_point[2], pivot_ref)
                    self.relative_tibia_points_list.append(relative_pos)
                elif len(self.tibia_list) == 2:
                    print("TLC (Tibia Lateral Condyle) point taken.")
                    self.tibia_list.append(pivot_ref)
                    relative_pos = TransformationUtils.get_relative_position(corner_point[2], pivot_ref)
                    self.relative_tibia_points_list.append(relative_pos)
                elif len(self.tibia_list) == 3:
                    print("Tuberosity point taken.")
                    self.tibia_list.append(pivot_ref)
                    relative_pos = TransformationUtils.get_relative_position(corner_point[2], pivot_ref)
                    self.relative_tibia_points_list.append(relative_pos)
                elif len(self.tibia_list) == 4:
                    print("PCL (Posterior Cruciate Ligament) point taken.")
                    self.tibia_list.append(pivot_ref)
                    relative_pos = TransformationUtils.get_relative_position(corner_point[2], pivot_ref)
                    self.relative_tibia_points_list.append(relative_pos)
                else: # len(self.tibia_list) == 5, for Ankle Center (AC)
                    print("Storing AC calc points.")
                    self.ac_fetcher_list.append(pivot_ref)
                    if len(self.ac_fetcher_list) == 2:
                        print("AC Finder Points taken. Calculating Ankle Center.")
                        self.tibia_list.append(TransformationUtils.find_ankle_center(self.ac_fetcher_list[0], self.ac_fetcher_list[1]))
                        relative_pos = TransformationUtils.get_relative_position(corner_point[2], pivot_ref)
                        self.relative_tibia_points_list.append(relative_pos)
                        self.ac_fetcher_list = [] # Reset for next use
                        print("Ankle Center Found.")
                print(f"Stored pivot_ref: {pivot_ref}")
                print(f"Current tibia_list: {self.tibia_list}")
        except Exception as e:
            print(f"Error in tibia_point_taker: {e}")

    def femur_point_taker(self, corner_point: list, joystick_obj, ct_side_selected: bool, T_ref_fem: np.ndarray):
        """
        Collects anatomical points for the femur based on joystick input.
        Args:
            corner_point (list): Current list of detected marker corners.
            joystick_obj: Pygame joystick object.
            ct_side_selected (bool): True if patient side is selected.
            T_ref_fem (np.ndarray): Transformation matrix for femur reference.
        """
        pointer_corners = corner_point[0]
        if (pointer_corners == self.zero_array).all() or (corner_point[3] == self.zero_array).all():
            return # Do not proceed if pointer or femur marker is not visible

        try:
            pivot_cam = self.pivot_point_obj.get_pivot_point_CamCoordSys(pointer_corners)
            pivot_ref = SQ_MARKER_NAVIGATION_SUPPORT_LIB.apply_tranform_to_points(pivot_cam, T_ref_fem)
            pivot_ref = pivot_ref[4,:]

            # This part is for Hip Center (HC) calculation, which uses `femur_first_frame`
            reference_femur = None
            if self.is_taking_hip_center and self.femur_first_frame is not None:
                # `getRelativeToMarker` expects 4x3, but `corner_point[3]` is 4x3.
                # `target_matrix` is 4x3.
                femur_reference = TransformationUtils.get_relative_to_marker(corner_point[3], corner_point[1], self.target_matrix)
                reference_femur = TransformationUtils.get_relative_to_marker(femur_reference, self.femur_first_frame, self.target_matrix)


            pygame.event.pump()
            current_time = pygame.time.get_ticks()
            button_a_current = joystick_obj.get_button(0)
            button_a_pressed = button_a_current and not self.prev_button_a
            self.prev_button_a = button_a_current

            button_allowed = button_a_pressed and (current_time - self.last_button_a_trigger >= 1000)

            if button_allowed and self.femur_first_frame is not None and len(self.femur_list) <= 6 and ct_side_selected:
                self.last_button_a_trigger = current_time

                if len(self.femur_list) == 0: # Hip Center (HC)
                    if not (corner_point[3] == self.zero_array).all() and not (corner_point[1] == self.zero_array).all() and reference_femur is not None:
                        self.hip_center_fetcher_print_list.append(np.asarray(reference_femur))
                        self.hip_center_fetcher_list.append(np.asarray(reference_femur.T)) # Expects 3xN for get_pivot_point
                        if len(self.hip_center_fetcher_list) == 5:
                            print("5 Hip Center points collected. Calculating Hip Center.")
                            pp = np.asarray(self.hip_center_fetcher_list)
                            hip_center = TransformationUtils.get_pivot_point(pp)["pv_point"]
                            self.femur_list.append(hip_center)
                            relative_pos = TransformationUtils.get_relative_position(corner_point[3], hip_center)
                            self.relative_femur_points_list.append(relative_pos)
                            self.hip_center_fetcher_list = []
                            self.is_taking_hip_center = False # Done with HC collection
                            print("Hip Center Found.")
                elif len(self.femur_list) == 1: # FKC (Femur Knee Center)
                    if not (pointer_corners == self.zero_array).all() and not (corner_point[3] == self.zero_array).all():
                        print("FKC point taken.")
                        self.femur_list.append(pivot_ref)
                        relative_pos = TransformationUtils.get_relative_position(corner_point[3], pivot_ref)
                        self.relative_femur_points_list.append(relative_pos)
                elif len(self.femur_list) == 2: # DMP (Distal Medial Condyle)
                    if not (pointer_corners == self.zero_array).all() and not (corner_point[3] == self.zero_array).all():
                        print("DMP point taken.")
                        self.femur_list.append(pivot_ref)
                        relative_pos = TransformationUtils.get_relative_position(corner_point[3], pivot_ref)
                        self.relative_femur_points_list.append(relative_pos)
                elif len(self.femur_list) == 3: # DLP (Distal Lateral Condyle)
                    if not (pointer_corners == self.zero_array).all() and not (corner_point[3] == self.zero_array).all():
                        print("DLP point taken.")
                        self.femur_list.append(pivot_ref)
                        relative_pos = TransformationUtils.get_relative_position(corner_point[3], pivot_ref)
                        self.relative_femur_points_list.append(relative_pos)
                elif len(self.femur_list) == 4: # Surgical Medial
                    if not (pointer_corners == self.zero_array).all() and not (corner_point[3] == self.zero_array).all():
                        print("Surgical Medial point taken.")
                        self.femur_list.append(pivot_ref)
                        relative_pos = TransformationUtils.get_relative_position(corner_point[3], pivot_ref)
                        self.relative_femur_points_list.append(relative_pos)
                elif len(self.femur_list) == 5: # Surgical Lateral
                    if not (pointer_corners == self.zero_array).all() and not (corner_point[3] == self.zero_array).all():
                        print("Surgical Lateral point taken.")
                        self.femur_list.append(pivot_ref)
                        relative_pos = TransformationUtils.get_relative_position(corner_point[3], pivot_ref)
                        self.relative_femur_points_list.append(relative_pos)
                print(f"Stored pivot_ref: {pivot_ref}")
                print(f"Current femur_list: {self.femur_list}")
        except Exception as e:
            print(f"Error in femur_point_taker: {e}")

    def tibia_point_referencing(self, corner_point: list):
        """
        Updates the absolute positions of collected tibia points based on current marker pose.
        Args:
            corner_point (list): Current list of detected marker corners.
        """
        if (corner_point[2] == self.zero_array).all():
            return # Cannot reference if tibia marker is not visible

        for index, tp in enumerate(self.relative_tibia_points_list):
            new_child_position = TransformationUtils.update_child_position(corner_point[2], tp)
            self.tibia_list[index] = new_child_position

    def femur_point_referencing(self, corner_point: list):
        """
        Updates the absolute positions of collected femur points based on current marker pose.
        Args:
            corner_point (list): Current list of detected marker corners.
        """
        if (corner_point[3] == self.zero_array).all():
            return # Cannot reference if femur marker is not visible

        for index, tp in enumerate(self.relative_femur_points_list):
            new_child_position = TransformationUtils.update_child_position(corner_point[3], tp)
            self.femur_list[index] = new_child_position

    def reset_tibia_points(self):
        """Resets all collected tibia points and related lists."""
        self.tibia_list = []
        self.relative_tibia_points_list = []
        self.ac_fetcher_list = []
        print("Tibia points reset.")

    def reset_femur_points(self):
        """Resets all collected femur points and related lists."""
        self.femur_first_frame = None
        self.femur_list = []
        self.relative_femur_points_list = []
        self.hip_center_fetcher_list = []
        self.hip_center_fetcher_print_list = []
        self.is_taking_hip_center = True
        print("Femur points reset.")


class MeasurementCalculator:
    """
    Performs clinical measurements (angles, distances, ROM) based on collected points.
    Relies on TransformationUtils and Planes_genrate_LIB.
    """
    def __init__(self):
        self.var_angle_combined = "N/A"
        self.var_flex_combined = "N/A"
        self.tmc_distance = 0
        self.tlc_distance = 0
        self.dmp_distance = 0
        self.dlp_distance = 0
        self.rom_angle = 0
        self.coronal_rom_angle = "N/A"
        self.tibia_cuts_calculated = False # Flag to ensure plane normals are calculated once
        self.femur_cuts_calculated = False # Flag to ensure plane normals are calculated once

        # Store plane normals after initial calculation
        self.sagittal_normal_tibia = None
        self.axial_normal_tibia = None
        self.coronal_normal_tibia = None
        self.sagittal_normal_femur = None
        self.axial_normal_femur = None
        self.coronal_normal_femur = None

    def tibia_verify_cuts(self, tibia_list: list, corner_point: list, zero_array: np.ndarray, vp_normal: np.ndarray, center_points: list, ct_side: str):
        """
        Calculates and updates tibia cut verification measurements (Coronal, Sagittal, Medial, Lateral).
        Args:
            tibia_list (list): List of collected tibia anatomical points.
            corner_point (list): Current list of detected marker corners.
            zero_array (np.ndarray): A zero array for marker visibility check.
            vp_normal (np.ndarray): Normal vector of the verification paddle plane.
            center_points (list): List of detected marker center points.
            ct_side (str): Patient side ("L" or "R").
        """
        if len(tibia_list) < 6 or (corner_point[1] == zero_array).all():
            self.var_angle_combined = "N/A"
            self.var_flex_combined = "N/A"
            self.tmc_distance = 0
            self.tlc_distance = 0
            return

        kc_point = tibia_list[0]
        tmc_point = tibia_list[1]
        tlc_point = tibia_list[2]
        tuberosity_point = tibia_list[3]
        pcl_point = tibia_list[4]
        ankle_center = tibia_list[5]

        mechanical_axis_vector = ankle_center - kc_point
        mechanical_axis_vector = mechanical_axis_vector / np.linalg.norm(mechanical_axis_vector)

        if not self.tibia_cuts_calculated:
            tuberosity_to_pcl = pcl_point - tuberosity_point
            tuberosity_to_pcl = tuberosity_to_pcl / np.linalg.norm(tuberosity_to_pcl)

            self.sagittal_normal_tibia = tuberosity_to_pcl
            # sagittal_plane = Plane(tuple(kc_point), normal_vector=self.sagittal_normal_tibia) # Not directly used after normal calc

            self.axial_normal_tibia = mechanical_axis_vector
            # axial_plane = Plane(tuple(kc_point), normal_vector=self.axial_normal_tibia) # Not directly used after normal calc

            self.coronal_normal_tibia = np.cross(self.sagittal_normal_tibia, self.axial_normal_tibia)
            self.coronal_normal_tibia = self.coronal_normal_tibia / np.linalg.norm(self.coronal_normal_tibia)
            # coronal_plane = Plane(tuple(kc_point), normal_vector=self.coronal_normal_tibia) # Not directly used after normal calc

            self.tibia_cuts_calculated = True
            print("Tibia anatomical planes calculated.")

        # Varus/Valgus calculation (in coronal plane using sagittal normal as axis of rotation)
        # Project mechanical axis and marker normal onto the plane orthogonal to sagittal normal
        var_angle = Planes_genrate_LIB.signed_angle(
            (Planes_genrate_LIB.project_on_plane(mechanical_axis_vector, self.sagittal_normal_tibia) - Planes_genrate_LIB.project_on_plane(-mechanical_axis_vector, self.sagittal_normal_tibia)),
            (Planes_genrate_LIB.project_on_plane(vp_normal, self.sagittal_normal_tibia) - Planes_genrate_LIB.project_on_plane(-vp_normal, self.sagittal_normal_tibia)),
            self.sagittal_normal_tibia
        )[0]

        # Anterior/Posterior calculation (in sagittal plane using coronal normal as axis of rotation)
        var_flex = Planes_genrate_LIB.signed_angle(
            (Planes_genrate_LIB.project_on_plane(mechanical_axis_vector, self.coronal_normal_tibia) - Planes_genrate_LIB.project_on_plane(-mechanical_axis_vector, self.coronal_normal_tibia)),
            (Planes_genrate_LIB.project_on_plane(vp_normal, self.coronal_normal_tibia) - Planes_genrate_LIB.project_on_plane(-vp_normal, self.coronal_normal_tibia)),
            self.coronal_normal_tibia
        )[0]

        self.tmc_distance = -int(Planes_genrate_LIB.point_plane_distance(tmc_point, vp_normal, center_points[1]))
        self.tlc_distance = -int(Planes_genrate_LIB.point_plane_distance(tlc_point, vp_normal, center_points[1]))

        # Determine naming convention based on CT side and angle values
        if var_angle < 0:
            var_angle_name = "Varus" if ct_side == "L" else "Valgus"
        else:
            var_angle_name = "Valgus" if ct_side == "L" else "Varus"

        if var_flex < 0:
            var_flex_name = "Ant" if ct_side == "L" else "Ant" # Original had "Ant" for both L/R negative
        else:
            var_flex_name = "Post" if ct_side == "L" else "Post" # Original had "Post" for both L/R positive

        self.var_angle_combined = f" {abs(var_angle):.0f} {var_angle_name}"
        self.var_flex_combined = f" {abs(var_flex):.0f} {var_flex_name}"

    def femur_verify_cuts(self, femur_list: list, corner_point: list, zero_array: np.ndarray, vp_normal: np.ndarray, center_points: list, ct_side: str):
        """
        Calculates and updates femur cut verification measurements (Coronal, Sagittal, Medial, Lateral).
        Args:
            femur_list (list): List of collected femur anatomical points.
            corner_point (list): Current list of detected marker corners.
            zero_array (np.ndarray): A zero array for marker visibility check.
            vp_normal (np.ndarray): Normal vector of the verification paddle plane.
            center_points (list): List of detected marker center points.
            ct_side (str): Patient side ("L" or "R").
        """
        if len(femur_list) < 6 or (corner_point[1] == zero_array).all():
            self.var_angle_combined = "N/A"
            self.var_flex_combined = "N/A"
            self.dmp_distance = 0
            self.dlp_distance = 0
            return

        hip_center = femur_list[0]
        knee_center = femur_list[1]
        dmp = femur_list[2]
        dlp = femur_list[3]
        surgical_medial = femur_list[4]
        surgical_lateral = femur_list[5]

        if not self.femur_cuts_calculated:
            self.coronal_normal_femur = np.array(surgical_lateral) - np.array(surgical_medial)
            self.coronal_normal_femur = self.coronal_normal_femur / np.linalg.norm(self.coronal_normal_femur)
            # coronal_plane = Plane(tuple(knee_center), normal_vector=self.coronal_normal_femur)

            self.axial_normal_femur = np.array(hip_center) - np.array(knee_center)
            self.axial_normal_femur = self.axial_normal_femur / np.linalg.norm(self.axial_normal_femur)
            # axial_plane = Plane(tuple(knee_center), normal_vector=self.axial_normal_femur)

            self.sagittal_normal_femur = np.cross(self.coronal_normal_femur, self.axial_normal_femur)
            self.sagittal_normal_femur = self.sagittal_normal_femur / np.linalg.norm(self.sagittal_normal_femur)
            # sagittal_plane = Plane(tuple(knee_center), normal_vector=self.sagittal_normal_femur)

            self.femur_cuts_calculated = True
            print("Femur anatomical planes calculated.")

        # Varus/Valgus calculation (in coronal plane using sagittal normal as axis of rotation)
        var_angle = Planes_genrate_LIB.signed_angle(
            (Planes_genrate_LIB.project_on_plane(self.axial_normal_femur, self.sagittal_normal_femur) - Planes_genrate_LIB.project_on_plane(-self.axial_normal_femur, self.sagittal_normal_femur)),
            (Planes_genrate_LIB.project_on_plane(vp_normal, self.sagittal_normal_femur) - Planes_genrate_LIB.project_on_plane(-vp_normal, self.sagittal_normal_femur)),
            self.sagittal_normal_femur
        )[0]

        # Ante/Post (Slope/Flexion/Extension) calculation (in sagittal plane using coronal normal as axis of rotation)
        var_flex = Planes_genrate_LIB.signed_angle(
            (Planes_genrate_LIB.project_on_plane(self.axial_normal_femur, self.coronal_normal_femur) - Planes_genrate_LIB.project_on_plane(-self.axial_normal_femur, self.coronal_normal_femur)),
            (Planes_genrate_LIB.project_on_plane(vp_normal, self.coronal_normal_femur) - Planes_genrate_LIB.project_on_plane(-vp_normal, self.coronal_normal_femur)),
            self.coronal_normal_femur
        )[0]

        self.dmp_distance = -int(Planes_genrate_LIB.point_plane_distance(dmp, vp_normal, center_points[1]))
        self.dlp_distance = -int(Planes_genrate_LIB.point_plane_distance(dlp, vp_normal, center_points[1]))

        if var_angle < 0:
            var_angle_name = "Varus" if ct_side == "L" else "Varus" # Original had "Varus" for both L/R negative
        else:
            var_angle_name = "Valgus" if ct_side == "L" else "Valgus" # Original had "Valgus" for both L/R positive

        if var_flex < 0:
            var_flex_name = "Flex" if ct_side == "L" else "Flex" # Original had "Ext" for L and "Ext" for R negative
        else:
            var_flex_name = "Ext" if ct_side == "L" else "Ext" # Original had "Flex" for L and "Flex" for R positive

        self.var_angle_combined = f"{abs(var_angle):.0f} {var_angle_name}"
        self.var_flex_combined = f" {abs(var_flex):.0f} {var_flex_name}"

    def calculate_rom(self, femur_hip: np.ndarray, femur_knee: np.ndarray, tibia_knee: np.ndarray, tibia_ankle: np.ndarray, ct_side: str):
        """
        Calculates Range of Motion (ROM) and Coronal ROM angle.
        Args:
            femur_hip (np.ndarray): Hip center point.
            femur_knee (np.ndarray): Femur knee center point.
            tibia_knee (np.ndarray): Tibia knee center point.
            tibia_ankle (np.ndarray): Tibia ankle center point.
            ct_side (str): Patient side ("L" or "R").
        """
        if self.sagittal_normal_femur is None:
            print("Sagittal normal for femur not calculated. Cannot compute ROM.")
            self.rom_angle = 0
            self.coronal_rom_angle = "N/A"
            return

        femur_vector = femur_knee - femur_hip
        tibia_vector = tibia_knee - tibia_ankle

        # Sagittal ROM
        rom_angle_raw = TransformationUtils.calculate_signed_angle(femur_vector, tibia_vector, self.sagittal_normal_femur)
        self.rom_angle = 180 - abs(rom_angle_raw) # Anatomical angle

        # Coronal ROM angle
        # This uses the sagittal_normal_femur as the axis for the coronal angle, which is consistent
        # with the original `Angle` function's usage for `coronal_Dir`.
        coronal_rom_angle_str, _ = TransformationUtils.Angle(femur_hip, femur_knee, tibia_knee, tibia_ankle, self.sagittal_normal_femur, ct_side)
        self.coronal_rom_angle = coronal_rom_angle_str


class GUI:
    """
    Manages the Tkinter GUI for displaying application status and camera feed.
    """
    def __init__(self, root_window: tk.Tk, on_closing_callback, toggle_camera_callback):
        self.root = root_window
        self.on_closing_callback = on_closing_callback
        self.toggle_camera_callback = toggle_camera_callback

        # GUI elements (initialized to None, created in _create_gui_elements)
        self.side_label = None
        self.object_label = None
        self.tibia_label = None
        self.pointer_label = None
        self.verification_label = None
        self.coronal_label = None
        self.sagittal_label = None
        self.medial_label = None
        self.lateral_label = None
        self.point_taken_label = None
        self.rom_label = None
        self.coronal_rom_angle_label = None
        self.cam_canvas = None
        self.show_camera = tk.BooleanVar(value=True) # State for camera display toggle
        self.camera_on_img = None
        self.camera_off_img = None
        self.camera_icon_label = None

        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()

        self._load_camera_icons()
        self._create_gui_elements()

    def _load_camera_icons(self):
        """Loads camera on/off icons from files."""
        try:
            self.camera_on_img = ImageTk.PhotoImage(Image.open("cam_on.png").resize((40, 30)))
            self.camera_off_img = ImageTk.PhotoImage(Image.open("cam_off.png").resize((40, 30)))
        except FileNotFoundError:
            print("Camera icons not found. Please ensure 'cam_on.png' and 'cam_off.png' are in the same directory.")
            # Fallback to dummy images if icons are not found
            self.camera_on_img = ImageTk.PhotoImage(Image.new('RGB', (40, 30), 'green'))
            self.camera_off_img = ImageTk.PhotoImage(Image.new('RGB', (40, 30), 'red'))

    def _create_gui_elements(self):
        """Creates and packs all Tkinter GUI elements."""
        self.root.title("Depth of Cut GUI")
        self.root.geometry(f"{self.screen_width}x{self.screen_height}")
        self.root.attributes('-fullscreen', True)
        self.root.state('zoomed')
        self.root.configure(bg='black')

        # Info Frame (Patient Side, Selected Object)
        info_frame = tk.Frame(self.root, bd=2, relief="solid", padx=10, pady=50, bg='black')
        info_frame.pack(fill="x", pady=5)

        normal_font = tkFont.Font(family="Arial", size=18, weight=tkFont.NORMAL)
        self.side_label = tk.Label(info_frame, text="Patient Side: N/A", font=normal_font, anchor="w", padx=self.screen_width/2.5, pady=-20, bg='black', fg='white')
        self.side_label.pack(fill="x", pady=5)

        self.object_label = tk.Label(info_frame, text="Selected Object: N/A", font=("Helvetica", 18), anchor="w", padx=self.screen_width/2.5, pady=-20, bg='black', fg='white')
        self.object_label.pack(fill="x", pady=5)

        # Left Frame (Marker Status, Point Taken Status, Camera Controls, Quit Button)
        left_frame = tk.Frame(self.root, relief="solid", padx=10, bg='black')
        left_frame.pack(side=tk.LEFT, fill=tk.Y, pady=10, padx=50)

        # Marker Status Frame
        marker_frame = tk.Frame(left_frame, bd=2, relief="solid", padx=10, pady=5, bg='black')
        marker_frame.pack(fill="x", pady=5)

        self.tibia_label = tk.Label(marker_frame, text="", font=("Helvetica", 18), width=30, height=2, bg='black', fg='white')
        self.tibia_label.pack(pady=5)

        self.pointer_label = tk.Label(marker_frame, text="", font=("Helvetica", 18), width=30, height=2, bg='black', fg='white')
        self.pointer_label.pack(pady=5)

        self.verification_label = tk.Label(marker_frame, text="", font=("Helvetica", 18), width=30, height=2, bg='black', fg='white')
        self.verification_label.pack(pady=5)

        # Point Taken Status Frame
        point_frame = tk.Frame(left_frame, padx=10, pady=10, bg='black')
        point_frame.pack(pady=5)

        self.point_taken_label = tk.Label(point_frame, text="Point taken", font=("Helvetica", 18), anchor="w", bg='#314d79', fg='white')
        self.point_taken_label.pack(pady=5)

        # Right Frame (Measurement Values)
        right_frame = tk.Frame(self.root, bg='black')
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, pady=10, padx=150)

        label_width = 15
        label_height = 1
        rom_label_width = 20

        self.coronal_label = tk.Label(right_frame, text="Coronal: N/A", font=("Helvetica", 20), anchor="w", width=label_width, height=label_height, bg='#314d79', fg='white')
        self.coronal_label.pack()

        self.sagittal_label = tk.Label(right_frame, text="Sagittal: N/A", font=("Helvetica", 20), anchor="w", width=label_width, height=label_height, bg='#314d79', fg='white')
        self.sagittal_label.pack()

        self.medial_label = tk.Label(right_frame, text="Medial: N/A", font=("Helvetica", 20), anchor="w", width=label_width, height=label_height, bg='#314d79', fg='white')
        self.medial_label.pack()

        self.lateral_label = tk.Label(right_frame, text="Lateral: N/A", font=("Helvetica", 20), anchor="w", width=label_width, height=label_height, bg='#314d79', fg='white')
        self.lateral_label.pack()

        self.rom_label = tk.Label(right_frame, text="ROM: N/A", font=("Helvetica", 20), anchor="w", width=rom_label_width, height=label_height, bg='#314d79', fg='white')
        self.rom_label.pack(pady=5, padx=5, fill="x")

        self.coronal_rom_angle_label = tk.Label(right_frame, text="Coronal ROM: N/A", font=("Helvetica", 20), anchor="w", width=rom_label_width, height=label_height, bg='#314d79', fg='white')
        self.coronal_rom_angle_label.pack(pady=5, padx=5, fill="x")

        # Camera Display Frame (at the bottom)
        camera_frame = tk.Frame(self.root, bd=2, relief="solid", pady=5, bg='black')
        camera_frame.pack(side=tk.BOTTOM, fill="both", expand=True)

        self.cam_canvas = tk.Label(camera_frame, bg='black')
        self.cam_canvas.pack(fill="both", expand=True)

        # Camera Control Frame (within left_frame)
        camera_control_frame = tk.Frame(left_frame, bg='black')
        camera_control_frame.pack(pady=20)

        camera_toggle_frame = tk.Frame(camera_control_frame, bg='black')
        camera_toggle_frame.pack()

        self.camera_icon_label = tk.Label(camera_toggle_frame, image=self.camera_on_img, bg='black', cursor='hand2')
        self.camera_icon_label.pack(side=tk.LEFT, padx=5)
        self.camera_icon_label.bind("<Button-1>", lambda event: self.toggle_camera_callback())

        camera_text_label = tk.Label(camera_toggle_frame, text="Camera Frame", font=("Helvetica", 14), bg='black', fg='white')
        camera_text_label.pack(side=tk.LEFT, padx=5)
        # camera_text_label.bind("<Button-1>", lambda event: self.toggle_camera_callback()) # Bind text label as well

        # Quit Button Frame (within left_frame, top-right within its parent)
        quit_frame = tk.Frame(left_frame, bg='black')
        quit_frame.pack(side=tk.LEFT, anchor='ne', padx=20, pady=10)

        quit_button = tk.Button(quit_frame, text="Quit (Q)", command=self.on_closing_callback, font=("Helvetica", 12, "bold"), bg='#FF4444', fg='white', activebackground='#CC0000', activeforeground='white', width=10, height=1, bd=0, padx=10, pady=5, cursor='hand2')
        quit_button.pack()
        self.root.bind('<q>', lambda event: self.on_closing_callback())
        self.root.bind('<Q>', lambda event: self.on_closing_callback())
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing_callback) # Handle window close button

    def update_gui(self, app_state: dict):
        """
        Updates the GUI elements based on the current application state.
        Args:
            app_state (dict): A dictionary containing all necessary state variables.
        """
        ct_side = app_state['ct_side']
        ct_side_selected = app_state['ct_side_selected']
        selected_object = app_state['selected_object']
        tibia_list_len = len(app_state['tibia_list'])
        femur_list_len = len(app_state['femur_list'])
        ac_fetcher_list_len = len(app_state['ac_fetcher_list'])
        hip_center_fetcher_list_len = len(app_state['hip_center_fetcher_list'])
        femur_first_frame_exists = app_state['femur_first_frame'] is not None
        is_rom_on = app_state['is_rom_on']
        show_reset_text = app_state['show_reset_text']
        corner_point = app_state['corner_point']
        zero_array = app_state['zero_array']

        # Update general info labels
        if ct_side_selected:
            self.side_label.config(text=f"Patient Side: {ct_side}")
        else:
            self.side_label.config(text="Patient Side: Please Select Side")

        self.object_label.config(text=f"Selected Part: {selected_object}")

        # Handle visibility of measurement labels based on ROM mode
        if is_rom_on and tibia_list_len >= 6 and femur_list_len >= 6:
            self.coronal_label.pack_forget()
            self.sagittal_label.pack_forget()
            self.medial_label.pack_forget()
            self.lateral_label.pack_forget()
            self.rom_label.pack()
            self.coronal_rom_angle_label.pack()
            self.point_taken_label.pack_forget() # Hide point taking instructions

            self.tibia_label.config(text="Tibia Marker", bg="green" if not (corner_point[2] == zero_array).all() else "red")
            self.tibia_label.pack()
            self.pointer_label.pack_forget() # Hide pointer label in ROM mode
            self.verification_label.config(text="Femur Marker", bg="green" if not (corner_point[3] == zero_array).all() else "red")
            self.verification_label.pack()

            self.rom_label.config(text=f"ROM: {int(app_state['rom_angle'])}°")
            self.coronal_rom_angle_label.config(text=f"Coronal ROM: {str(app_state['coronal_rom_angle'])}")

        else: # Not in ROM mode or points not collected for ROM
            self.rom_label.pack_forget()
            self.coronal_rom_angle_label.pack_forget()
            self.coronal_label.pack()
            self.sagittal_label.pack()
            self.medial_label.pack()
            self.lateral_label.pack()
            self.point_taken_label.pack() # Show point taking instructions

            # Update marker status and point taking instructions based on selected object
            if selected_object == "tibia":
                self.tibia_label.config(text="Tibia Marker", bg="green" if not (corner_point[2] == zero_array).all() else "red")
                self.pointer_label.pack() # Show pointer for point taking
                
                if tibia_list_len == 0:
                    self.point_taken_label.config(text="Take Point: Tibia Knee Center")
                    self.verification_label.pack_forget()
                elif tibia_list_len == 1:
                    self.point_taken_label.config(text="Take Point: Tibia Medial Condyle")
                    self.verification_label.pack_forget()
                elif tibia_list_len == 2:
                    self.point_taken_label.config(text="Take Point: Tibia Lateral Condyle")
                    self.verification_label.pack_forget()
                elif tibia_list_len == 3:
                    self.point_taken_label.config(text="Take Point: Tibia Tuberosity")
                    self.verification_label.pack_forget()
                elif tibia_list_len == 4:
                    self.point_taken_label.config(text="Take Point: PCL")
                    self.verification_label.pack_forget()
                elif tibia_list_len == 5:
                    if ac_fetcher_list_len == 0:
                        self.point_taken_label.config(text="Take Point: Medial Malleolus")
                    elif ac_fetcher_list_len == 1:
                        self.point_taken_label.config(text="Take Point: Lateral Malleolus")
                    self.verification_label.pack_forget()
                elif tibia_list_len == 6:
                    self.point_taken_label.config(text="Tibia Points Taken")
                    self.verification_label.pack_forget()

                # Marker visibility for tibia mode
                self.pointer_label.config(text="Pointer Marker", bg="green" if not (corner_point[0] == zero_array).all() else "red")
                self.verification_label.config(text="Verification Marker", bg="green" if not (corner_point[1] == zero_array).all() else "red")
                self.verification_label.pack() # Ensure verification marker is shown for cuts

                # Update measurement values for tibia
                self.coronal_label.config(text=f"Coronal: {app_state['var_angle_combined']}")
                self.sagittal_label.config(text=f"Sagittal: {app_state['var_flex_combined']}")
                self.medial_label.config(text=f"Medial: {app_state['tmc_distance']:.0f} mm")
                self.lateral_label.config(text=f"Lateral: {app_state['tlc_distance']:.0f} mm")

            elif selected_object == "femur":
                self.tibia_label.config(text="Femur Marker", bg="green" if not (corner_point[3] == zero_array).all() else "red")
                self.verification_label.pack() # Show verification marker (used as stable marker)
                self.pointer_label.pack() # Show pointer for point taking

                if not femur_first_frame_exists:
                    self.point_taken_label.config(text="Capture First frame of femur")
                    self.verification_label.config(text="Stable Marker", bg="green" if not (corner_point[1] == zero_array).all() else "red")
                    self.pointer_label.pack_forget() # Hide pointer until first frame is captured
                elif femur_list_len == 0 and femur_first_frame_exists:
                    self.point_taken_label.config(text=f"Femur Hip Center Points :{hip_center_fetcher_list_len + 1}")
                    self.verification_label.config(text="Stable Marker", bg="green" if not (corner_point[1] == zero_array).all() else "red")
                    self.pointer_label.pack_forget() # Hide pointer during HC collection
                elif femur_list_len == 1:
                    self.point_taken_label.config(text="Take Point: Femur Knee Center")
                elif femur_list_len == 2:
                    self.point_taken_label.config(text="Take Point: Femur Distal Medial Condyle")
                elif femur_list_len == 3:
                    self.point_taken_label.config(text="Take Point: Femur Distal Lateral Condyle")
                elif femur_list_len == 4:
                    self.point_taken_label.config(text="Take Point: Femur Surgical Medial Condyle")
                elif femur_list_len == 5:
                    self.point_taken_label.config(text="Take Point: Femur Surgical Lateral Condyle")
                elif femur_list_len == 6:
                    self.point_taken_label.config(text="Femur Points Taken")
                
                # Marker visibility for femur mode
                self.pointer_label.config(text="Pointer Marker", bg="green" if not (corner_point[0] == zero_array).all() else "red")
                self.verification_label.config(text="Verification Marker", bg="green" if not (corner_point[1] == zero_array).all() else "red")
                self.verification_label.pack() # Ensure verification marker is shown for cuts

                # Update measurement values for femur
                self.coronal_label.config(text=f"Coronal: {app_state['var_angle_combined']}")
                self.sagittal_label.config(text=f"Slope: {app_state['var_flex_combined']}")
                self.medial_label.config(text=f"Medial: {app_state['dmp_distance']:.0f} mm")
                self.lateral_label.config(text=f"Lateral: {app_state['dlp_distance']:.0f} mm")

        # Handle reset text display
        if show_reset_text:
            self.point_taken_label.config(text="Resetting...")
            self.verification_label.pack_forget() # Hide verification during reset message

        self.root.update_idletasks()
        self.root.update()

    def update_camera_feed(self, frame: np.ndarray, show_camera_flag: bool):
        """
        Updates the camera feed display in the GUI.
        Args:
            frame (np.ndarray): The camera frame (BGR format).
            show_camera_flag (bool): True if camera feed should be displayed.
        """
        if show_camera_flag:
            try:
                frame_width = self.cam_canvas.winfo_width()
                frame_height = self.cam_canvas.winfo_height()

                if frame_width > 1 and frame_height > 1:
                    # Calculate target dimensions maintaining aspect ratio and minimum size
                    frame_aspect = frame.shape[1] / frame.shape[0]
                    window_aspect = frame_width / frame_height

                    if window_aspect > frame_aspect:
                        new_height = frame_height
                        new_width = int(new_height * frame_aspect)
                    else:
                        new_width = frame_width
                        new_height = int(new_width / frame_aspect)
                    
                    # Ensure minimum size
                    min_height = int(self.screen_height * 0.3) # Adjusted for better scaling
                    min_width = int(self.screen_width * 0.3)
                    new_width = max(new_width, min_width)
                    new_height = max(new_height, min_height)

                    display_frame = cv2.resize(frame, (new_width, new_height))
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(display_frame)
                    imgtk = ImageTk.PhotoImage(image=img)

                    self.cam_canvas.imgtk = imgtk
                    self.cam_canvas.configure(image=imgtk)
            except Exception as e:
                print(f"Error updating camera feed: {str(e)}")
        else:
            self.cam_canvas.configure(image='')


class Application:
    """
    Main application class orchestrating all components.
    """
    
    def __init__(self):
        # Configuration
        self.system_config = {
            "aruco_ids": [8, 5, 11, 13], # Pointer, Verification, Tibia, Femur
            "aruco_size": 50, # Marker size in mm
            "aruco_dictionary": "DICT_4X4_50"
        }
        self.zero_array = np.zeros((4,3)) # Used for checking if a marker is detected (all zeros)
        self.target_matrix = np.array([ # Virtual marker corners for transformations
            [-30.0,30.0,0],
            [30.0,30.0,0],
            [30.0,-30.0,0],
            [-30.0,-30.0,0]
        ])

        # Initialize Pygame for joystick input
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            root_temp = tk.Tk()
            root_temp.withdraw()
            messagebox.showerror("Joystick Error", "No joystick/controller detected. Please connect a joystick and restart the application.")
            sys.exit()
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"Connected to joystick: {self.joystick.get_name()}")

        # Initialize Tkinter root window and GUI component
        self.root = tk.Tk()
        self.gui = GUI(self.root, self.on_closing, self.toggle_camera_display)

        # Initialize external dependencies (placeholders for now)
        # These imports assume `main1.py` and `CAMERA_CAPTURE_MODULE_MAIN.py` exist
        # and provide the expected `data` and `ComeraCapture_RealSense` classes.
        try:
            from main1 import data
            from CAMERA_CAPTURE_MODULE_MAIN import ComeraCapture_RealSense
            self.data_instance = data()
            self.camera_obj = ComeraCapture_RealSense()
            self.data_instance.obj2 = self.camera_obj # Assuming this connection is needed for data flow
        except ImportError as e:
            print(f"Error importing external modules: {e}")
            print("Please ensure 'main1.py' and 'CAMERA_CAPTURE_MODULE_MAIN.py' are correctly set up and accessible.")
            sys.exit(1)

        # Initialize core application components
        # Load pivot point calibration data
        try:
            pointer_corner_points = np.array(np.load('pv_corner_points_18_4_1_Small_pointer.npy'))
        except FileNotFoundError:
            print("Error: 'pv_corner_points_18_4_1_Small_pointer.npy' not found.")
            print("Please ensure the pivot point calibration file is in the same directory.")
            sys.exit(1)

        virtual_sq = SQ_MARKER_NAVIGATION_SUPPORT_LIB.get_vertual_SQ_coordinates(self.system_config['aruco_size'])
        pivot_point_cam = SQ_MARKER_NAVIGATION_SUPPORT_LIB.Comput_Aruco_Pivot_Point(pointer_corner_points)
        pivot_def = np.vstack((virtual_sq, pivot_point_cam))
        self.pivot_point_obj = SQ_MARKER_PIVOT_OBJ(pivot_def) # Object for pivot point calculations

        self.marker_tracker = MarkerTracker(self.system_config, self.camera_obj)
        self.point_manager = PointManager(self.system_config, self.pivot_point_obj)
        self.measurement_calculator = MeasurementCalculator()

        # Application state variables (moved from global scope)
        self.selected_object = "tibia" # Default selected object: "tibia" or "femur"
        self.ct_side = "" # Patient side: "L" (Left) or "R" (Right)
        self.ct_side_selected = False # Flag to ensure side is selected before proceeding
        self.is_rom_on = False # Flag for Range of Motion mode
        self.show_reset_text = False # Flag to display "Resetting..." message
        self.reset_start_time = 0 # Timestamp for reset message display duration

        # Input cooldowns (to prevent rapid triggering)
        self.last_button_b_trigger = 0 # For ROM toggle
        self.last_trigger_time = 0 # For single point removal (Y-hat)
        self.last_bumper_time = 0 # For side selection
        self.bumper_cooldown = 1000 # milliseconds

        self.only_at_first_time_femur_capture = True # Flag for capturing femur's first frame

        # Current frame data (updated each loop iteration)
        self.corner_point = [self.zero_array] * 4 # Stores 4x3 arrays of marker corners
        self.center_points = [np.zeros(3)] * 4 # Stores 1x3 arrays of marker centers
        self.vp_normal = None # Normal vector of the verification paddle plane

        # Transformation matrices for reference markers
        self.T_ref_tib = None
        self.T_ref_fem = None

    def toggle_camera_display(self):
        """Toggles the camera feed display on/off."""
        self.gui.toggle_camera_callback() # Delegate to GUI class

    def on_closing(self):
        """Handles application shutdown, including confirmation dialog and cleanup."""
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
            print("Closing application...")
            self.root.quit()
            self.root.destroy()
            cv2.destroyAllWindows()
            pygame.quit()
            sys.exit()

    def handle_input(self):
        """Processes joystick and keyboard inputs to update application state."""
        pygame.event.pump() # Process pending pygame events
        joystick = self.joystick
        current_time = pygame.time.get_ticks()

        # Get joystick button states
        button_a = joystick.get_button(0)
        button_b = joystick.get_button(1)
        button_x = joystick.get_button(2)
        button_y = joystick.get_button(3)
        left_bumper = joystick.get_button(4)
        right_bumper = joystick.get_button(5)
        back_button = joystick.get_button(6)
        x_axis_hat, y_axis_hat = joystick.get_hat(0) # D-pad input

        # 1. Patient Side Selection (Left Bumper / Right Bumper)
        if not self.ct_side_selected:
            if (left_bumper or right_bumper) and (current_time - self.last_bumper_time >= self.bumper_cooldown):
                if left_bumper:
                    self.ct_side = "L"
                elif right_bumper:
                    self.ct_side = "R"
                self.ct_side_selected = True
                self.last_bumper_time = current_time
                print(f"Patient Side Selected: {self.ct_side}")

        # 2. Object Selection (Button X for Femur, Button Y for Tibia)
        if button_x:
            self.is_rom_on = False # Exit ROM mode when selecting an object
            self.selected_object = "femur"
            self.point_manager.is_taking_hip_center = True # Reset HC collection for femur
            self.only_at_first_time_femur_capture = True # Allow first frame capture again
            self.measurement_calculator.femur_cuts_calculated = False # Reset femur plane calculation flag
            print("Selected Object: Femur")
        elif button_y:
            self.is_rom_on = False # Exit ROM mode when selecting an object
            self.selected_object = "tibia"
            self.measurement_calculator.tibia_cuts_calculated = False # Reset tibia plane calculation flag
            print("Selected Object: Tibia")

        # 3. Toggle ROM Mode (Button B)
        # Use point_manager's prev_button_a as a general 'prev_button_pressed' for joystick A,
        # but for Button B, we need a separate state to avoid conflicts.
        button_b_current = joystick.get_button(1)
        # A simple state for Button B's previous state
        if not hasattr(self, '_prev_button_b_state'):
            self._prev_button_b_state = False
        
        button_b_pressed_edge = button_b_current and not self._prev_button_b_state
        self._prev_button_b_state = button_b_current # Update previous state for next iteration

        if button_b_pressed_edge and (current_time - self.last_button_b_trigger >= 1000):
            self.is_rom_on = not self.is_rom_on
            # If ROM is ON, set selected_object to "none" to indicate a different mode
            self.selected_object = "none" if self.is_rom_on else ("femur" if len(self.point_manager.femur_list) > 0 else "tibia")
            self.last_button_b_trigger = current_time
            print(f"ROM Mode: {'ON' if self.is_rom_on else 'OFF'}")

        # 4. Capture Femur First Frame (Button A and specific conditions)
        # This is the initial frame for femur measurements, often a stable marker position.
        if (button_a and self.selected_object == "femur" and self.only_at_first_time_femur_capture and
            not (self.corner_point[1] == self.zero_array).all() and self.ct_side_selected):
            # Capture femur's current position relative to the stable verification marker
            femur_reference = TransformationUtils.get_relative_to_marker(self.corner_point[3], self.corner_point[1], self.target_matrix)
            self.point_manager.femur_first_frame = femur_reference
            print("Femur First Frame Captured.")
            self.only_at_first_time_femur_capture = False # Only capture once

        # 5. Reset Points (D-pad Right)
        if x_axis_hat == 1: # D-pad Right
            self.show_reset_text = True # Trigger "Resetting..." message
            self.reset_start_time = time.time() # Record time for message duration

            if self.selected_object == "femur":
                self.point_manager.reset_femur_points()
                self.only_at_first_time_femur_capture = True # Allow re-capture of first frame
                self.measurement_calculator.femur_cuts_calculated = False # Reset plane calculation flag
                print("Femur points reset.")
            elif self.selected_object == "tibia":
                self.point_manager.reset_tibia_points()
                self.measurement_calculator.tibia_cuts_calculated = False # Reset plane calculation flag
                print("Tibia points reset.")

        # 6. Remove Last Point (D-pad Down)
        if y_axis_hat == 1: # D-pad Down
            if current_time - self.last_trigger_time >= 1000: # Cooldown for point removal
                self.last_trigger_time = current_time

                if self.selected_object == "femur":
                    if len(self.point_manager.femur_list) == 1 and not self.point_manager.is_taking_hip_center:
                        # If only FKC is taken and not in HC collection, reset all femur points
                        self.point_manager.reset_femur_points()
                        self.only_at_first_time_femur_capture = True
                    elif len(self.point_manager.femur_list) > 0 and not self.point_manager.is_taking_hip_center:
                        # Remove last collected point if not in HC collection
                        removed_point = self.point_manager.femur_list.pop()
                        self.point_manager.relative_femur_points_list.pop()
                        print(f"Removed femur point: {removed_point}")
                    elif len(self.point_manager.femur_list) == 0 and len(self.point_manager.hip_center_fetcher_list) > 0:
                        # Remove last HC collection point
                        removed_hc_point = self.point_manager.hip_center_fetcher_list.pop()
                        self.point_manager.hip_center_fetcher_print_list.pop()
                        print(f"Removed Hip Center collection point: {removed_hc_point}")

                elif self.selected_object == "tibia":
                    if len(self.point_manager.tibia_list) > 0 and len(self.point_manager.ac_fetcher_list) == 0:
                        # Remove last collected point (KC, TMC, TLC, Tuberosity, PCL)
                        removed_point = self.point_manager.tibia_list.pop()
                        self.point_manager.relative_tibia_points_list.pop()
                        print(f"Removed tibia point: {removed_point}")
                    elif len(self.point_manager.ac_fetcher_list) > 0:
                        # Remove last AC collection point
                        removed_ac_point = self.point_manager.ac_fetcher_list.pop()
                        print(f"Removed Ankle Center collection point: {removed_ac_point}")

        # 7. Quit Application (Back Button on Joystick or 'q' key)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_q] or back_button:
            self.on_closing()

    def run(self):
        """Main application loop."""
        while True:
            # 1. Get Marker Data
            left_frame, self.corner_point, self.center_points, exact_matches, is_marker_detected = self.marker_tracker.get_marker_data()

            # 2. Update Reference Marker Transformations (T_ref)
            # These transformations are used to bring pointer points into the respective bone's coordinate system.
            if self.selected_object == "tibia":
                if not (self.corner_point[2] == self.zero_array).all():
                    self.T_ref_tib = SQ_MARKER_NAVIGATION_SUPPORT_LIB.GET_ARUCO_VREF_TRANSFORM(self.corner_point[2], self.system_config['aruco_size'])
                else:
                    self.T_ref_tib = None # Clear if marker not visible
            elif self.selected_object == "femur":
                if not (self.corner_point[3] == self.zero_array).all():
                    self.T_ref_fem = SQ_MARKER_NAVIGATION_SUPPORT_LIB.GET_ARUCO_VREF_TRANSFORM(self.corner_point[3], self.system_config['aruco_size'])
                else:
                    self.T_ref_fem = None # Clear if marker not visible

            # 3. Update Verification Paddle Normal (vp_normal)
            # This normal is crucial for calculating angles and distances relative to the verification plane.
            if not (self.corner_point[1] == self.zero_array).all(): # If verification marker is visible
                if self.selected_object == "tibia" and not (self.corner_point[2] == self.zero_array).all():
                    # Transform verification paddle corners relative to the tibia marker
                    vp_corners = TransformationUtils.get_relative_to_marker(self.corner_point[1], self.corner_point[2], self.target_matrix)
                    self.center_points[1] = np.mean(vp_corners, axis=0) # Update center of verification marker
                    self.vp_normal = Planes_genrate_LIB.calculate_plane_normal(vp_corners) # Calculate normal
                elif self.selected_object == "femur" and not (self.corner_point[3] == self.zero_array).all():
                    # Transform verification paddle corners relative to the femur marker
                    vp_corners = TransformationUtils.get_relative_to_marker(self.corner_point[1], self.corner_point[3], self.target_matrix)
                    self.center_points[1] = np.mean(vp_corners, axis=0)
                    self.vp_normal = Planes_genrate_LIB.calculate_plane_normal(vp_corners)
                else:
                    self.vp_normal = None # Clear if relevant bone marker is not visible
            else:
                self.vp_normal = None # Clear if verification marker is not visible

            # 4. Handle User Input
            self.handle_input()

            # 5. Point Collection and Referencing
            if self.selected_object == "tibia":
                # Only attempt to take points if pointer and tibia markers are visible and T_ref_tib is available
                if not (self.corner_point[0] == self.zero_array).all() and not (self.corner_point[2] == self.zero_array).all() and self.T_ref_tib is not None:
                    self.point_manager.tibia_point_taker(self.corner_point, self.joystick, self.ct_side_selected, self.T_ref_tib)
                # Always attempt to reference points if tibia marker is visible
                self.point_manager.tibia_point_referencing(self.corner_point)
                # Calculate measurements if enough points are collected and verification marker is visible
                if self.vp_normal is not None:
                    self.measurement_calculator.tibia_verify_cuts(self.point_manager.tibia_list, self.corner_point, self.zero_array, self.vp_normal, self.center_points, self.ct_side)
            elif self.selected_object == "femur":
                # Only attempt to take points if pointer and femur markers are visible and T_ref_fem is available
                if not (self.corner_point[0] == self.zero_array).all() and not (self.corner_point[3] == self.zero_array).all() and self.T_ref_fem is not None:
                    self.point_manager.femur_point_taker(self.corner_point, self.joystick, self.ct_side_selected, self.T_ref_fem)
                # Always attempt to reference points if femur marker is visible
                self.point_manager.femur_point_referencing(self.corner_point)
                # Calculate measurements if enough points are collected and verification marker is visible
                if self.vp_normal is not None:
                    self.measurement_calculator.femur_verify_cuts(self.point_manager.femur_list, self.corner_point, self.zero_array, self.vp_normal, self.center_points, self.ct_side)

            # 6. ROM Calculation (if in ROM mode and all required points are collected)
            if self.is_rom_on and len(self.point_manager.tibia_list) >= 6 and len(self.point_manager.femur_list) >= 6:
                try:
                    # Transform all required points to a common coordinate system (e.g., femur marker's current frame)
                    # This is crucial for accurate ROM calculation.
                    # Original code used `getRelativeToPoint` with `corner_point[3]` (femur marker) as the reference.
                    # We need to ensure `virtual_to_camera` and `get_relative_to_point` are consistent.
                    
                    # First, transform relative points (stored in marker's local frame) to camera frame
                    New_femur_hip_cam = TransformationUtils.virtual_to_camera(self.point_manager.relative_femur_points_list[0], self.corner_point[3])
                    New_femur_knee_cam = TransformationUtils.virtual_to_camera(self.point_manager.relative_femur_points_list[1], self.corner_point[3])
                    New_tibia_knee_cam = TransformationUtils.virtual_to_camera(self.point_manager.relative_tibia_points_list[0], self.corner_point[2])
                    New_tibia_ankle_cam = TransformationUtils.virtual_to_camera(self.point_manager.relative_tibia_points_list[3], self.corner_point[2])

                    # Then, transform points from camera frame to the current femur marker's frame
                    if not (self.corner_point[3] == self.zero_array).all():
                        New_femur_hip = TransformationUtils.get_relative_to_point(New_femur_hip_cam, self.corner_point[3])
                        New_femur_knee = TransformationUtils.get_relative_to_point(New_femur_knee_cam, self.corner_point[3])
                        New_tibia_knee = TransformationUtils.get_relative_to_point(New_tibia_knee_cam, self.corner_point[3])
                        New_tibia_ankle = TransformationUtils.get_relative_to_point(New_tibia_ankle_cam, self.corner_point[3])

                        self.measurement_calculator.calculate_rom(
                            New_femur_hip, New_femur_knee, New_tibia_knee, New_tibia_ankle, self.ct_side
                        )
                    else:
                        print("Cannot calculate ROM: Femur marker not visible.")
                        self.measurement_calculator.rom_angle = 0
                        self.measurement_calculator.coronal_rom_angle = "N/A"

                except Exception as e:
                    print(f"ROM Calculation Error: {str(e)}")
                    self.measurement_calculator.rom_angle = 0
                    self.measurement_calculator.coronal_rom_angle = "N/A"
            else:
                self.measurement_calculator.rom_angle = 0
                self.measurement_calculator.coronal_rom_angle = "N/A"

            # 7. Update GUI
            # Prepare a dictionary of current application state to pass to the GUI
            app_state_for_gui = {
                'ct_side': self.ct_side,
                'ct_side_selected': self.ct_side_selected,
                'selected_object': self.selected_object,
                'tibia_list': self.point_manager.tibia_list,
                'femur_list': self.point_manager.femur_list,
                'ac_fetcher_list': self.point_manager.ac_fetcher_list,
                'hip_center_fetcher_list': self.point_manager.hip_center_fetcher_list,
                'femur_first_frame': self.point_manager.femur_first_frame,
                'is_rom_on': self.is_rom_on,
                'show_reset_text': self.show_reset_text,
                'corner_point': self.corner_point,
                'zero_array': self.zero_array,
                'var_angle_combined': self.measurement_calculator.var_angle_combined,
                'var_flex_combined': self.measurement_calculator.var_flex_combined,
                'tmc_distance': self.measurement_calculator.tmc_distance,
                'tlc_distance': self.measurement_calculator.tlc_distance,
                'dmp_distance': self.measurement_calculator.dmp_distance,
                'dlp_distance': self.measurement_calculator.dlp_distance,
                'rom_angle': self.measurement_calculator.rom_angle,
                'coronal_rom_angle': self.measurement_calculator.coronal_rom_angle,
            }
            self.gui.update_gui(app_state_for_gui)
            self.gui.update_camera_feed(left_frame, self.gui.show_camera.get())

            # 8. Manage "Resetting..." message display duration
            if self.show_reset_text and (time.time() - self.reset_start_time >= 2):
                self.show_reset_text = False

            # Control loop speed
            time.sleep(0.05)

# Main execution block
if __name__ == '__main__':
    data_instance = data()
    app = Application()
    app.run()
