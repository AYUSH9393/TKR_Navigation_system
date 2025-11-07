# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 10:59:25 2022

@author: admin
"""

import numpy as np
import cv2
#import tkinter as tk
#from tkinter import messagebox
#import threading
import time
import queue
#from collections import deque
#import requests
#from io import BytesIO
from picamera2 import Picamera2

# class CameraCapture_USB():
#     def __init__(self, CAMERA_CONFIG_OBJ=None):
#         rtsp_url = "rtsp://10.42.0.1:8554/cam"
#         rtmp_url = "rtmp://10.42.0.1:1935/cam"
#         self.cap = cv2.VideoCapture(rtsp_url)
        
#         # Critical: Set buffer size to 1 BEFORE other settings
#         self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
#         # Set resolution and FPS
#         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
#         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#         self.cap.set(cv2.CAP_PROP_FPS, 30)
        
#         # Codec optimization for RTSP
#         self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H','2','6','4'))
        
#         # Additional RTSP optimizations
#         self.cap.set(cv2.CAP_PROP_POS_MSEC, 0)  # Seek to latest frame
        
#         # Store the target resolution
#         self.target_width = 2560
#         self.target_height = 720
        
#         # Frame buffering for threading
#         self.latest_frame = None
#         self.frame_lock = threading.Lock()
#         self.capture_thread = None
#         self.running = False
        
#         # Start background capture thread
#         self.start_capture_thread()
        
#         print("Camera initialized successfully")
    
#     def start_capture_thread(self):
#         """Start background thread for continuous frame capture"""
#         self.running = True
#         self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
#         self.capture_thread.start()
    
#     def _capture_frames(self):
#         """Background thread function to continuously capture frames"""
#         while self.running:
#             ret, frame = self.cap.read()
#             if ret:
#                 with self.frame_lock:
#                     self.latest_frame = frame
#             time.sleep(0.001)  # Small delay to prevent CPU overload
    
#     def get_latest_frame(self):
#         """Get the most recent frame from buffer"""
#         with self.frame_lock:
#             return self.latest_frame.copy() if self.latest_frame is not None else None
            
#     def get_IR_FRAME_SET(self):
#         L_img = np.empty((0,0))
#         R_img = np.empty((0,0))
        
#         # Get the latest frame from background thread
#         frame = self.get_latest_frame()
        
#         if frame is None:
#             print("Warning: No frame available from camera")
#             return False, L_img, R_img, None
        
#         ret = True
        
#         # Resize frame if needed
#         if frame.shape[1] != self.target_width or frame.shape[0] != self.target_height:
#             frame = cv2.resize(frame, (self.target_width, self.target_height))
        
#         # Split the frame into left and right images
#         split_point = self.target_width // 2
#         L_img = frame[:, :split_point]
#         R_img = frame[:, split_point:]
        
#         return ret, L_img, R_img, frame
    
#     def release(self):
#         """Clean up resources"""
#         self.running = False
#         if self.capture_thread:
#             self.capture_thread.join(timeout=1.0)
#         self.cap.release()

class CameraCapture_Pi:
    def __init__(self, resolution=(1280, 400), use_yuv=True, fps=30):
        """
        High-speed stereo capture using PiCamera2.
        Splits left/right frames without background threads.
        """
        self.cam = Picamera2()
        self.use_yuv = use_yuv
        self.fps = fps

        fmt = "YUV420" if use_yuv else "RGB888"
        cfg = self.cam.create_preview_configuration(
            main={"format": fmt, "size": resolution},
            buffer_count=4
        )

        frame_duration = int(1e6 / fps)
        cfg["controls"]["FrameDurationLimits"] = (frame_duration, frame_duration)

        self.cam.configure(cfg)
        self.cam.start()
        print(f"PiCamera started at {resolution}, {fps} FPS using {fmt}")

    def get_IR_FRAME_SET(self):
        """Capture and immediately return (ret, left_img, right_img, full_gray)."""
        try:
            frame = self.cam.capture_array()
            if frame is None or frame.size == 0:
                return False, np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0))
                
            # Convert to grayscale efficiently
            if self.use_yuv:
                gray = cv2.cvtColor(frame, cv2.COLOR_YUV2GRAY_I420)
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Split into left / right images
            h, w = gray.shape[:2]
            mid = w // 2
            left_img = gray[:, :mid]
            right_img = gray[:, mid:]
            return True, left_img, right_img, gray

        except Exception as e:
            print(f"[Camera] capture error: {e}")
            return False, np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0))

    def release(self):
        """Stop camera safely."""
        try:
            self.cam.stop()
        except Exception as e:
            print(f"Error stopping camera: {e}")
        print("Camera released.")


"""
class CameraCapture_USB():
    def __init__(self, CAMERA_CONFIG_OBJ=None):
        # Use RTMP URL instead of RTSP
        rtmp_url = "rtmp://10.42.0.1:1935/cam"
        #rtsp_url = "rtsp://10.42.0.1:8554/cam"  # Keep as backup
        
        # Try RTMP first, fallback to RTSP if needed
        self.cap = cv2.VideoCapture(rtmp_url)
        
        # # Check if RTMP connection successful
        # if not self.cap.isOpened():
        #     print("RTMP connection failed, trying RTSP as fallback...")
        #     self.cap = cv2.VideoCapture(rtsp_url)
        #     if not self.cap.isOpened():
        #         raise Exception("Failed to connect to both RTMP and RTSP streams")
        # else:
        #     print("Successfully connected to RTMP stream")
        
        # Critical: Set buffer size to 1 BEFORE other settings for low latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Set resolution and FPS
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        #self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # RTMP-specific optimizations
        # Note: RTMP typically works better with these codecs
        #self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H','2','6','4'))
        
        # Get the actual camera resolution after initialization
        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # RTMP doesn't need POS_MSEC setting like RTSP
        # Instead, rely on low buffer size for latest frames
        
        # Store the target resolution
        # self.target_width = 1280
        # self.target_height = 400
        
        # Frame buffering for threading
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.capture_thread = None
        self.running = False
        
        # Statistics for monitoring
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        
        # Add FPS tracking variables
        self.fps = 0
        self.frame_time = time.time()
        self.fps_update_interval = 1.0 
        
        # Start background capture thread
        self.start_capture_thread()
        
        print("Camera initialized successfully")
        
    
    def start_capture_thread(self):
        
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()
        print("Background capture thread started")
    
    def _capture_frames(self):
        
        consecutive_failures = 0
        max_failures = 30  # Allow 30 consecutive failures before giving up
        
        while self.running:
            ret, frame = self.cap.read()
            
            if ret:
                consecutive_failures = 0
                
                # Update FPS calculation
                current_time = time.time()
                time_diff = current_time - self.frame_time
                if time_diff >= self.fps_update_interval:
                    self.fps = self.fps_counter / time_diff
                    self.fps_counter = 0
                    self.frame_time = current_time
                
                with self.frame_lock:
                    self.latest_frame = frame
                    self.frame_count += 1
                    self.fps_counter += 1
                    
            else:
                consecutive_failures += 1
                print(f"Frame capture failed (attempt {consecutive_failures})")
                
                if consecutive_failures >= max_failures:
                    print("Too many consecutive failures, stopping capture thread")
                    self.running = False
                    break
                
                time.sleep(0.1)  # Wait before retrying
                continue
            
            # Small delay to prevent CPU overload, but keep it minimal for RTMP
            time.sleep(0.001)
    
    def get_latest_frame(self):
        
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def get_IR_FRAME_SET(self):
        
        L_img = np.empty((0,0))
        R_img = np.empty((0,0))
        
        # Get the latest frame from background thread
        frame = self.get_latest_frame()
        
        if frame is None:
            print("Warning: No frame available from camera")
            return False, L_img, R_img, None
        
        ret = True
        #cv2.imshow("Camera Frame", frame)  # Show the frame for debugging
        #scale of frame
        #print(f"Frame shape: {frame.shape}")
        
        
        #print(f"Received frame: {actual_width}x{actual_height}")
        
        # Resize frame if needed
        # if actual_width != self.target_width or actual_height != self.target_height:
        #     print(f"Resizing from {actual_width}x{actual_height} to {self.target_width}x{self.target_height}")
        #     frame = cv2.resize(frame, (self.target_width, self.target_height))
        
        # Check actual frame dimensions (no resizing needed now)
        actual_height, actual_width = frame.shape[:2]
        #print(f"Received frame: {actual_width}x{actual_height}")
        
        # Split the frame into left and right images for stereo camera
        # Using actual width instead of hardcoded target_width
        split_point = actual_width // 2  # Split at center
        L_img = frame[:, :split_point]      # Left half
        R_img = frame[:, split_point:]      # Right half

        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(L_img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2, cv2.LINE_AA)
        return ret, L_img, R_img ,frame
    
    def get_frame_info(self):
        
        frame = self.get_latest_frame()
        if frame is not None:
            height, width, channels = frame.shape
            return {
                'width': width,
                'height': height, 
                'channels': channels,
                'total_frames': self.frame_count
            }
        return None
    
    def is_opened(self):
        
        return self.cap.isOpened() and self.running
    
    def reconnect(self):
        
        print("Attempting to reconnect to camera...")
        self.release()
        time.sleep(1)  # Brief pause
        
        try:
            self.__init__()  # Reinitialize
            return True
        except Exception as e:
            print(f"Reconnection failed: {e}")
            return False
    
    def release(self):
        
        print("Releasing camera resources...")
        self.running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
            if self.capture_thread.is_alive():
                print("Warning: Capture thread did not stop cleanly")
        
        if self.cap:
            self.cap.release()
        
        print("Camera resources released")
"""
