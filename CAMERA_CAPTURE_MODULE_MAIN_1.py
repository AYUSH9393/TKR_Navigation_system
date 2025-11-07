# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 10:59:25 2022

@author: admin
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import tkinter as tk
from tkinter import messagebox
import threading
import time
from collections import deque
import requests
from io import BytesIO

class CameraCapture_USB():
    def __init__(self, CAMERA_CONFIG_OBJ=None):
        # RTMP URL configuration - modify host/port as needed
        self.host = "10.42.0.1"  # Your MediaMTX host
        self.rtmp_port = 1935    # Your RTMP port
        self.stream_name = "cam" # Your stream name
        
        rtmp_url = f"rtmp://{self.host}:{self.rtmp_port}/{self.stream_name}"
        print(f"Connecting to RTMP stream: {rtmp_url}")
        
        # Initialize VideoCapture with RTMP
        self.cap = cv2.VideoCapture(rtmp_url)
        
        # Check if connection successful
        if not self.cap.isOpened():
            print("RTMP connection failed!")
            raise Exception(f"Failed to connect to RTMP stream: {rtmp_url}")
        else:
            print("Successfully connected to RTMP stream")
        
        # Critical: Set buffer size to 1 BEFORE other settings for low latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Optional: Try to set resolution and FPS (may not work with all streams)
        # try:
        #     self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        #     self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        #     self.cap.set(cv2.CAP_PROP_FPS, 30)
        # except:
        #     print("Could not set custom resolution, using stream defaults")
        
        # Get the actual stream resolution after initialization
        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Stream resolution: {self.actual_width}x{self.actual_height}")
        
        # Frame buffering for threading
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.capture_thread = None
        self.running = False
        
        # Statistics for monitoring
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        
        # FPS tracking variables
        self.fps = 0
        self.frame_time = time.time()
        self.fps_update_interval = 1.0 
        
        # Reconnection handling
        self.last_reconnect_attempt = 0
        self.reconnect_interval = 5.0  # seconds between reconnection attempts
        
        # Start background capture thread
        self.start_capture_thread()
        
        print("Camera initialized successfully")
        
    def start_capture_thread(self):
        """Start background thread for continuous frame capture"""
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()
        print("Background capture thread started")
    
    def _capture_frames(self):
        """Background thread function to continuously capture frames"""
        consecutive_failures = 0
        max_failures = 60  # Allow more failures for network streams
        
        while self.running:
            try:
                # Check if stream is still open
                if not self.cap or not self.cap.isOpened():
                    print("Stream disconnected, attempting reconnection...")
                    self._attempt_reconnect()
                    time.sleep(1)
                    continue
                
                ret, frame = self.cap.read()
                
                if ret and frame is not None:
                    consecutive_failures = 0
                    
                    # Update FPS calculation
                    current_time = time.time()
                    time_diff = current_time - self.frame_time
                    if time_diff >= self.fps_update_interval:
                        self.fps = self.fps_counter / time_diff if time_diff > 0 else 0
                        self.fps_counter = 0
                        self.frame_time = current_time
                    
                    with self.frame_lock:
                        self.latest_frame = frame.copy()
                        self.frame_count += 1
                        self.fps_counter += 1
                        
                else:
                    consecutive_failures += 1
                    
                    if consecutive_failures >= max_failures:
                        print(f"Too many consecutive failures ({consecutive_failures}), attempting reconnection...")
                        self._attempt_reconnect()
                        consecutive_failures = 0
                        time.sleep(2)
                        continue
                    
                    # Brief wait before retry
                    time.sleep(0.1)
                    continue
                
            except Exception as e:
                print(f"Error in capture thread: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    print("Too many errors, attempting reconnection...")
                    self._attempt_reconnect()
                    consecutive_failures = 0
                time.sleep(0.1)
                continue
            
            # Small delay to prevent CPU overload, but keep it minimal for RTMP
            time.sleep(0.001)
    
    def _attempt_reconnect(self):
        """Attempt to reconnect to the RTMP stream"""
        current_time = time.time()
        if current_time - self.last_reconnect_attempt < self.reconnect_interval:
            return  # Too soon to reconnect
        
        self.last_reconnect_attempt = current_time
        print("Attempting to reconnect to RTMP stream...")
        
        # Release current connection
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Wait a bit before reconnecting
        time.sleep(1)
        
        try:
            # Recreate RTMP connection
            rtmp_url = f"rtmp://{self.host}:{self.rtmp_port}/{self.stream_name}"
            self.cap = cv2.VideoCapture(rtmp_url)
            
            if self.cap.isOpened():
                # Test frame reading
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    print("✓ Reconnection successful")
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reset buffer size
                    return True
                else:
                    print("✗ Reconnection failed - no frames")
                    self.cap.release()
                    self.cap = None
            else:
                print("✗ Reconnection failed - cannot open stream")
                
        except Exception as e:
            print(f"✗ Reconnection error: {e}")
        
        return False
    
    def get_latest_frame(self):
        """Get the most recent frame from buffer"""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def get_IR_FRAME_SET(self):
        """Get stereo frame set (left and right images)"""
        L_img = np.empty((0,0))
        R_img = np.empty((0,0))
        
        # Get the latest frame from background thread
        frame = self.get_latest_frame()
        
        if frame is None:
            print("Warning: No frame available from camera")
            return False, L_img, R_img, None
        
        ret = True
        
        # Check actual frame dimensions
        actual_height, actual_width = frame.shape[:2]
        
        # Split the frame into left and right images for stereo camera
        # Using actual width instead of hardcoded target_width
        split_point = actual_width // 2  # Split at center
        L_img = frame[:, :split_point]      # Left half
        R_img = frame[:, split_point:]      # Right half

        # Add FPS and stream info text
        fps_text = f"FPS: {self.fps:.1f} | RTMP"
        cv2.putText(L_img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Add connection info
        connection_text = f"{self.host}:{self.rtmp_port}/{self.stream_name}"
        cv2.putText(L_img, connection_text, (10, L_img.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        
        return ret, L_img, R_img, frame
    
    def get_frame_info(self):
        """Get current frame information for debugging"""
        frame = self.get_latest_frame()
        if frame is not None:
            height, width, channels = frame.shape
            return {
                'width': width,
                'height': height, 
                'channels': channels,
                'total_frames': self.frame_count,
                'fps': self.fps,
                'stream_url': f"rtmp://{self.host}:{self.rtmp_port}/{self.stream_name}",
                'running': self.running
            }
        return None
    
    def is_opened(self):
        """Check if camera is still connected"""
        return self.cap and self.cap.isOpened() and self.running
    
    def reconnect(self):
        """Attempt to reconnect to camera (public method)"""
        print("Manual reconnection requested...")
        return self._attempt_reconnect()
    
    def release(self):
        """Clean up resources"""
        print("Releasing camera resources...")
        self.running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=3.0)
            if self.capture_thread.is_alive():
                print("Warning: Capture thread did not stop cleanly")
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        print("Camera resources released")