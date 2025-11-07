# -*- coding: utf-8 -*-
"""
MediaMTX Camera Access for Raspberry Pi 5
Support for RTSP, RTMP, and WebRTC streams
"""

import cv2
import numpy as np
import threading
import time
import tkinter as tk
from tkinter import messagebox
import requests
from urllib.parse import urlparse

class MediaMTXCameraCapture:
    def __init__(self, stream_config=None):
        """
        Initialize camera capture via MediaMTX
        stream_config: dict with stream configuration or None for auto-detection
        """
        self.cap = None
        self.camera_initialized = False
        self.current_stream_url = None
        self.stream_type = None
        
        # Default MediaMTX configuration (adjust as needed)
        self.default_config = {
            'host': 'localhost',  # or 'raspberrypi.local' or IP address
            'rtsp_port': 8554,
            'rtmp_port': 1935,
            'webrtc_port': 8889,
            'api_port': 9997,
            'stream_name': 'cam',  # Your stream name in MediaMTX
        }
        
        # Update with user config
        if stream_config:
            self.default_config.update(stream_config)
        
        # Threading components
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.capture_thread = None
        self.running = False
        
        # Statistics
        self.frame_count = 0
        self.fps_counter = 0
        self.fps = 0
        self.frame_time = time.time()
        self.fps_update_interval = 1.0
        self.last_reconnect_attempt = 0
        self.reconnect_interval = 5.0  # seconds
        
        # Initialize connection
        self._initialize_stream()
        
        if self.camera_initialized:
            self.start_capture_thread()
    
    def _get_stream_urls(self):
        """Generate possible stream URLs based on configuration"""
        config = self.default_config
        host = config['host']
        stream_name = config['stream_name']
        
        urls = {
            'rtsp': f"rtsp://{host}:{config['rtsp_port']}/{stream_name}",
            'rtmp': f"rtmp://{host}:{config['rtmp_port']}/{stream_name}",
            # WebRTC requires different handling, not directly supported by OpenCV
        }
        
        return urls
    
    def _check_mediamtx_status(self):
        """Check if MediaMTX is running and stream is available"""
        try:
            api_url = f"http://{self.default_config['host']}:{self.default_config['api_port']}/v3/paths/list"
            response = requests.get(api_url, timeout=2)
            
            if response.status_code == 200:
                data = response.json()
                stream_name = self.default_config['stream_name']
                
                # Check if our stream exists and has publishers
                for item in data.get('items', []):
                    if item.get('name') == stream_name:
                        ready = item.get('ready', False)
                        num_readers = item.get('numReaders', 0)
                        print(f"Stream '{stream_name}': Ready={ready}, Readers={num_readers}")
                        return ready
                        
                print(f"Stream '{stream_name}' not found in MediaMTX")
                return False
            else:
                print(f"MediaMTX API returned status: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"Cannot connect to MediaMTX API: {e}")
            return False
    
    def _initialize_stream(self):
        """Initialize stream connection with fallback options"""
        print("Initializing MediaMTX stream connection...")
        
        # Check MediaMTX status first
        if not self._check_mediamtx_status():
            print("Warning: MediaMTX stream may not be ready")
        
        urls = self._get_stream_urls()
        
        # Try different protocols in order of preference
        protocols = ['rtsp', 'rtmp']  # RTSP usually has lower latency
        
        for protocol in protocols:
            if protocol not in urls:
                continue
                
            url = urls[protocol]
            print(f"Trying {protocol.upper()}: {url}")
            
            try:
                # Configure OpenCV VideoCapture based on protocol
                if protocol == 'rtsp':
                    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                    # RTSP optimizations
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                elif protocol == 'rtmp':
                    cap = cv2.VideoCapture(url)
                    # RTMP optimizations
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                if cap.isOpened():
                    # Test frame reading
                    print("Testing frame capture...")
                    ret, test_frame = cap.read()
                    
                    if ret and test_frame is not None:
                        print(f"✓ Successfully connected via {protocol.upper()}")
                        print(f"Frame shape: {test_frame.shape}")
                        
                        self.cap = cap
                        self.current_stream_url = url
                        self.stream_type = protocol
                        self.camera_initialized = True
                        
                        # Get stream properties
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        print(f"Stream properties: {width}x{height} @ {fps:.1f}fps")
                        return
                    else:
                        print(f"✗ {protocol.upper()} opened but no frames received")
                        cap.release()
                else:
                    print(f"✗ Cannot open {protocol.upper()} stream")
                    
            except Exception as e:
                print(f"✗ Error with {protocol.upper()}: {e}")
                if 'cap' in locals():
                    cap.release()
                continue
        
        print("Failed to connect to any MediaMTX stream")
        self._show_error_dialog("Cannot connect to MediaMTX streams. Check if MediaMTX is running and camera is streaming.")
    
    def _show_error_dialog(self, message):
        """Show error dialog"""
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("MediaMTX Camera Error", message)
            root.destroy()
        except:
            print(f"Error: {message}")
    
    def start_capture_thread(self):
        """Start background thread for continuous frame capture"""
        if not self.camera_initialized:
            return
            
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()
        print(f"Background capture thread started for {self.stream_type.upper()} stream")
    
    def _capture_frames(self):
        """Background thread function to continuously capture frames"""
        consecutive_failures = 0
        max_failures = 60  # Allow more failures for network streams
        
        while self.running:
            try:
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
            
            # Small delay to prevent CPU overload
            time.sleep(0.001)
    
    def _attempt_reconnect(self):
        """Attempt to reconnect to the stream"""
        current_time = time.time()
        if current_time - self.last_reconnect_attempt < self.reconnect_interval:
            return  # Too soon to reconnect
        
        self.last_reconnect_attempt = current_time
        print("Attempting to reconnect to MediaMTX stream...")
        
        # Release current connection
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.camera_initialized = False
        
        # Wait a bit before reconnecting
        time.sleep(1)
        
        # Try to reinitialize
        self._initialize_stream()
    
    def get_latest_frame(self):
        """Get the most recent frame from buffer"""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def get_IR_FRAME_SET(self):
        """
        Get stereo frame set (left and right images)
        Compatible with your existing interface
        """
        L_img = np.empty((0,0))
        R_img = np.empty((0,0))
        
        if not self.camera_initialized:
            print("Camera not initialized")
            return False, L_img, R_img, None
        
        # Get the latest frame from background thread
        frame = self.get_latest_frame()
        
        if frame is None:
            print("Warning: No frame available from MediaMTX stream")
            return False, L_img, R_img, None
        
        # Get frame dimensions
        actual_height, actual_width = frame.shape[:2]
        aspect_ratio = actual_width / actual_height
        
        # Determine if this is a stereo camera setup
        if aspect_ratio > 2.0:  # Wide aspect ratio suggests side-by-side stereo
            # Split the frame into left and right images
            split_point = actual_width // 2
            L_img = frame[:, :split_point]      # Left half
            R_img = frame[:, split_point:]      # Right half
            
            info_text = f"Stereo - {self.stream_type.upper()}"
        else:
            # Single camera - use same frame for both sides
            L_img = frame.copy()
            R_img = frame.copy()
            
            # Add indicator text to distinguish left/right
            cv2.putText(R_img, "Right (Copy)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 255), 2, cv2.LINE_AA)
            info_text = f"Single - {self.stream_type.upper()}"
        
        # Add status information to left image
        status_text = f"FPS: {self.fps:.1f} | {info_text}"
        cv2.putText(L_img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Add connection info
        connection_text = f"MediaMTX: {self.default_config['host']}"
        cv2.putText(L_img, connection_text, (10, L_img.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        
        return True, L_img, R_img, frame
    
    def get_stream_info(self):
        """Get current stream information"""
        if not self.camera_initialized:
            return None
        
        frame = self.get_latest_frame()
        info = {
            'stream_url': self.current_stream_url,
            'stream_type': self.stream_type,
            'host': self.default_config['host'],
            'stream_name': self.default_config['stream_name'],
            'fps': self.fps,
            'total_frames': self.frame_count,
            'running': self.running,
            'initialized': self.camera_initialized
        }
        
        if frame is not None:
            height, width = frame.shape[:2]
            channels = frame.shape[2] if len(frame.shape) > 2 else 1
            info.update({
                'width': width,
                'height': height,
                'channels': channels
            })
        
        return info
    
    def is_opened(self):
        """Check if stream is connected and running"""
        return (self.camera_initialized and 
                self.cap and 
                self.cap.isOpened() and 
                self.running)
    
    def reconnect(self):
        """Force reconnection to stream"""
        print("Manual reconnection requested...")
        self._attempt_reconnect()
        return self.camera_initialized
    
    def release(self):
        """Clean up resources"""
        print(f"Releasing MediaMTX {self.stream_type} stream...")
        self.running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=3.0)
            if self.capture_thread.is_alive():
                print("Warning: Capture thread did not stop cleanly")
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.camera_initialized = False
        print("MediaMTX stream resources released")


# Configuration examples for different setups
def get_local_config():
    """Configuration for local MediaMTX on same Pi"""
    return {
        'host': 'localhost',
        'rtsp_port': 8554,
        'rtmp_port': 1935,
        'stream_name': 'cam'
    }

def get_remote_config(pi_ip):
    """Configuration for MediaMTX on remote Pi"""
    return {
        'host': pi_ip,  # e.g., '192.168.1.100' or 'raspberrypi.local'
        'rtsp_port': 8554,
        'rtmp_port': 1935,
        'stream_name': 'cam'
    }

def get_custom_config(host, stream_name, rtsp_port=8554, rtmp_port=1935):
    """Custom configuration"""
    return {
        'host': host,
        'rtsp_port': rtsp_port,
        'rtmp_port': rtmp_port,
        'stream_name': stream_name
    }


# Example usage and testing
if __name__ == "__main__":
    print("MediaMTX Camera Capture Test")
    print("=" * 40)
    
    # Use local configuration by default
    config = get_local_config()
    
    # Or use remote configuration:
    # config = get_remote_config('192.168.1.100')
    
    # Or custom configuration:
    # config = get_custom_config('10.42.0.1', 'mycam')
    
    print(f"Connecting to MediaMTX at {config['host']}")
    print(f"Stream name: {config['stream_name']}")
    
    try:
        # Create camera capture object
        camera = MediaMTXCameraCapture(config)
        
        if not camera.is_opened():
            print("Failed to connect to MediaMTX stream")
            print("Make sure:")
            print("1. MediaMTX is running")
            print("2. Camera is streaming to MediaMTX")
            print("3. Network connection is working")
            exit(1)
        
        print("Stream connected successfully!")
        stream_info = camera.get_stream_info()
        print(f"Stream info: {stream_info}")
        
        print("\nStarting frame capture (press 'q' to quit, 'r' to reconnect)...")
        
        frame_display_interval = 30  # Show info every N frames
        
        while True:
            ret, L_img, R_img, full_frame = camera.get_IR_FRAME_SET()
            
            if ret:
                # Display the frames
                cv2.imshow('Left Image (MediaMTX)', L_img)
                cv2.imshow('Right Image (MediaMTX)', R_img)
                
                # Optional: show full frame
                if full_frame is not None:
                    # Resize for display if too large
                    display_frame = full_frame
                    if display_frame.shape[1] > 1280:
                        scale = 1280 / display_frame.shape[1]
                        new_width = int(display_frame.shape[1] * scale)
                        new_height = int(display_frame.shape[0] * scale)
                        display_frame = cv2.resize(display_frame, (new_width, new_height))
                    
                    cv2.imshow('Full Stream', display_frame)
                
                # Print stream info periodically
                if camera.frame_count % frame_display_interval == 0:
                    info = camera.get_stream_info()
                    if info:
                        print(f"Frames: {info['total_frames']}, "
                              f"FPS: {info['fps']:.1f}, "
                              f"Resolution: {info.get('width', '?')}x{info.get('height', '?')}, "
                              f"Stream: {info['stream_type'].upper()}")
            else:
                print("Failed to get frame from MediaMTX")
                time.sleep(0.1)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("Manual reconnection...")
                camera.reconnect()
        
        # Cleanup
        cv2.destroyAllWindows()
        camera.release()
        print("Test completed successfully!")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        cv2.destroyAllWindows()
        if 'camera' in locals():
            camera.release()
    
    except Exception as e:
        print(f"Error during testing: {e}")
        cv2.destroyAllWindows()
        if 'camera' in locals():
            camera.release()