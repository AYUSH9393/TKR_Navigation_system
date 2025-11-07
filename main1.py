#from save_images import SaveImages
import configparser
import pickle
from camera_calibrate import StereoCalibration
import numpy as np
import math
from KEYPOINTS_DETECT_LIB import SterioParameter
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2
from CAMERA_CAPTURE_MODULE_MAIN import CameraCapture_Pi as CameraCapture_USB
#from CAMERA_CAPTURE_MODULE_MAIN import CameraCapture_USB
import time
import csv
from collections import deque
import threading
from queue import Queue, Empty
# config = configparser.ConfigParser()
# config.read('config.ini')
from fast_aruco import ArUco
def calculate_perpendicular_angle(corner_points):
    # Convert corner points to NumPy array
    corner_points = np.array(corner_points, dtype=np.float32)
    #print("Corner_point:", corner_points)

    # Compute the normal vector of the ArUco marker plane using cross product
    v1 = corner_points[1] - corner_points[0]
    print("v1:",v1)
    v2 = corner_points[2] - corner_points[0]
    normal_vector = np.cross(v1, v2)

    # Normalize the normal vector
    normal_vector /= np.linalg.norm(normal_vector)

    return normal_vector

def angle_between_two_vectors(v1, v2):
     
    # Calculate the angle between two vectors
    angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)),-1.0,1.0))
 
    # Return the angle in degrees
    return np.degrees(angle)

def calculate_distance(point1, point2):
    x1 = point1[0]
    x2 = point2[0]
    y1 = point1[1]
    y2 = point2[1]
    z1 = point1[2]
    z2 = point2[2]
    d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    return d

def angle_between_two_aruco(d1,d2):
    d1 = np.array(d1)
    d2 = np.array(d2)

    print(d1.shape)
    n1 = np.cross(d1[2,:]-d1[1,:],d1[2,:]-d1[3,:])    
    n2 = np.cross(d2[2,:]-d2[1,:],d2[2,:]-d2[3,:])   
    v1_u = n1 / np.linalg.norm(n1)
    v2_u = n2 / np.linalg.norm(n2)
    angle_rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    angle_degree = np.rad2deg(angle_rad)
    return angle_degree


# def CalibrateCamera():
#     path = config['PATHS']['CalibrationImagePath']
#     img_format = config['IMAGE_FORMAT']['CalibrationImageFormat']

#     # calibrating camera and saving calibration file
#     cal = StereoCalibration(filepath=path, img_format=img_format)
#     try :
#         with open('cal_10X7_A4_1.pkl' , 'wb' )as fp:
#             pickle.dump(cal,fp)
#     except:
#         print("Unable to save")

class data:
    def __init__(self):
        self.ids = [8, 5, 11, 13]
        self.StereoObj = SterioParameter("cal_10X7_A4_16.pkl")
        self.obj = ArUco()
        self.obj.set_tool_vector(self.ids)
        self.obj2 = CameraCapture_USB(resolution=(2560, 720))
        self.corner_points_history = deque(maxlen=6)
        self.center_points_history = deque(maxlen=6)
    
    def send_data(self):
        while True:
            ret, left, right, frames = self.obj2.get_IR_FRAME_SET()
            corner_tool_ret, corner_point = self.obj.get_corner_world_points(self.StereoObj, left, right)
            center_tool_ret, center_points = self.obj.get_center_world_points(self.StereoObj, left, right)

            # Handle None values in corner_point
            corner_point_processed = []
            for cp in corner_point:
                if cp is None or cp.size == 0:  # Handle None or empty arrays
                    cp = np.zeros((4, 3))  # Use zeros for untracked markers
                corner_point_processed.append(cp)

            # Convert corner_point to a single 3D NumPy array (n_markers x 4 x 3)
            corner_point_matrix = np.array(corner_point_processed)

            # Handle None values in center_points
            center_points_processed = []
            for cp in center_points:
                if cp is None or cp.size == 0:  # Handle None or empty arrays
                    cp = np.zeros((1, 3))  # Use zeros for untracked markers
                center_points_processed.append(cp.flatten())

            # Convert center_points to a 2D NumPy array (n_markers x 3)
            center_points_matrix = np.vstack(center_points_processed)

            # Store the current frame's data
            self.corner_points_history.append(corner_point_matrix)
            self.center_points_history.append(center_points_matrix)

            # Keep only the last 7 frames
            if len(self.corner_points_history) > 6:
                self.corner_points_history.pop(0)
            if len(self.center_points_history) > 6:
                self.center_points_history.pop(0)

            # Ensure that we always have a valid median_corner_points and median_center_points
            if len(self.corner_points_history) >= 6:
                median_corner_points = np.mean(self.corner_points_history, axis=0)
                median_center_points = np.mean(self.center_points_history, axis=0)
                #print("median values")
            else:
                median_corner_points = corner_point_matrix
                median_center_points = center_points_matrix

            return median_corner_points, median_center_points, frames

class data_1_o:
    def __init__(self):
        self.ids = [8, 5, 11, 13]
        self.StereoObj = SterioParameter("cal_10X7_A4_27.pkl")
        self.obj = ArUco()
        self.obj.set_tool_vector(self.ids)
        self.obj2 = CameraCapture_USB(resolution=(2560, 720))
        self.corner_points_history = deque(maxlen=7)
        self.center_points_history = deque(maxlen=7)
        
        # Threading setup for better performance
        self.frame_queue = Queue(maxsize=2)
        self.processed_data_queue = Queue(maxsize=7)
        self.capture_thread = None
        self.processing_thread = None
        self.running = False
        
        # Locks for thread safety
        self.history_lock = threading.Lock()
        self.camera_lock = threading.Lock()  # Lock for camera access
        
        # Error handling
        self.camera_initialized = True
        
        self.EXPECTED_MARKER_SIZE = 50.0  # mm
        self.MARKER_SIZE_TOLERANCE = 2.0  # mm

    def start_threads(self):
        '''Start the capture and processing threads'''
        self.running = True
        
        # Start frame capture thread
        self.capture_thread = threading.Thread(target=self._frame_capture_thread, daemon=True)
        self.capture_thread.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_thread, daemon=True)
        self.processing_thread.start()
        
    def stop_threads(self):
        '''Stop all threads gracefully'''
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
    
    def _frame_capture_thread(self):
        '''Continuously capture frames in a separate thread'''
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while self.running:
            try:
                ret, left, right, frames = self.obj2.get_IR_FRAME_SET()
                if ret:
                    consecutive_errors = 0  # Reset error counter on success
                    frame_data = (left, right, frames)
                    
                    # If queue is full, remove old frame and add new one
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except Empty:
                            pass
                    
                    try:
                        self.frame_queue.put(frame_data, timeout=0.01)
                    except:
                        # Queue is full, skip this frame
                        continue
                else:
                    # Camera didn't return a frame, small delay
                    time.sleep(0.01)
                        
            except Exception as e:
                consecutive_errors += 1
                print(f"Error in frame capture thread: {e}")
                
                # If too many consecutive errors, increase delay
                if consecutive_errors > max_consecutive_errors:
                    print("Too many consecutive camera errors, using longer delay...")
                    time.sleep(0.1)
                else:
                    time.sleep(0.01)
                
                # Reset camera connection if too many errors
                if consecutive_errors > 20:
                    print("Attempting to reset camera connection...")
                    try:
                        # Reinitialize camera object
                        self.obj2 = CameraCapture_USB(CAMERA_CONFIG_OBJ=None)
                        consecutive_errors = 0
                    except Exception as reset_error:
                        print(f"Failed to reset camera: {reset_error}")
                        time.sleep(1.0)
    
    def _processing_thread(self):
        '''Process frames in a separate thread'''
        while self.running:
            try:
                # Get frame from queue with timeout
                left, right, frames = self.frame_queue.get(timeout=1.0)
                
                # Process the frame
                corner_tool_ret, corner_point = self.obj.get_corner_world_points(self.StereoObj, left, right)
                center_tool_ret, center_points = self.obj.get_center_world_points(self.StereoObj, left, right)
                
                # Handle None values in corner_point
                corner_point_processed = []
                for cp in corner_point:
                    if cp is None or cp.size == 0:
                        cp = np.zeros((4, 3))
                    corner_point_processed.append(cp)
                
                # Convert corner_point to a single 3D NumPy array
                corner_point_matrix = np.array(corner_point_processed)
                
                # Handle None values in center_points
                center_points_processed = []
                for cp in center_points:
                    if cp is None or cp.size == 0:
                        cp = np.zeros((1, 3))
                    center_points_processed.append(cp.flatten())
                
                # Convert center_points to a 2D NumPy array
                center_points_matrix = np.vstack(center_points_processed)
                
                # Thread-safe update of history
                with self.history_lock:
                    self.corner_points_history.append(corner_point_matrix)
                    self.center_points_history.append(center_points_matrix)
                
                # Put processed data in output queue
                processed_data = (corner_point_matrix, center_points_matrix, frames)
                
                if self.processed_data_queue.full():
                    try:
                        self.processed_data_queue.get_nowait()
                    except Empty:
                        pass
                
                try:
                    self.processed_data_queue.put(processed_data, timeout=0.01)
                except:
                    continue
                    
            except Empty:
                # No frame available, continue
                continue
            except Exception as e:
                print(f"Error in processing thread: {e}")
                continue
    
    def get_latest_data(self, timeout=1.0):
        '''Get the latest processed data'''
        try:
            return self.processed_data_queue.get(timeout=timeout)
        except Empty:
            return None, None, None
    
    def get_median_data(self):
        '''Get median values from history (thread-safe)'''
        with self.history_lock:
            if len(self.corner_points_history) >= 7:
                median_corner_points = np.mean(list(self.corner_points_history), axis=0)
                median_center_points = np.mean(list(self.center_points_history), axis=0)
                print("median values")
                return median_corner_points, median_center_points
            elif len(self.corner_points_history) > 0:
                # Return the latest data if we don't have enough history
                return self.corner_points_history[-1], self.center_points_history[-1]
            else:
                return None, None
    
    def send_data(self, return_median=True):
        '''
        Fetches processed marker data and performs validation.
        Only returns data if marker sizes are valid.
        Returns empty arrays if validation fails.
        '''
        # Start threads if not running
        if not self.running:
            self.start_threads()
            time.sleep(0.1)  # allow threads to warm up

        # Step 1: Fetch data (either median or latest)
        if return_median:
            corner_point, center_point = self.get_median_data()
            _, _, frames = self.get_latest_data()
        else:
            corner_point, center_point, frames = self.get_latest_data()

        # Step 2: Handle missing data
        if corner_point is None or center_point is None:
            empty_corners = np.zeros((len(self.ids), 4, 3))
            empty_centers = np.zeros((len(self.ids), 3))
            return empty_corners, empty_centers, None

        # Step 3: Validate marker sizes (only for detected markers)
        valid_sizes = True
        marker_sizes = []
        detected_count = 0

        for i in range(len(self.ids)):
            # Check if marker is detected
            if len(corner_point) > i and corner_point[i] is not None and corner_point[i].size != 0:
                # Check if it's not just zeros
                if not np.allclose(corner_point[i], 0):
                    detected_count += 1
                    try:
                        # Calculate side lengths of each detected marker
                        dists = [
                            calculate_distance(corner_point[i][0], corner_point[i][1]),
                            calculate_distance(corner_point[i][1], corner_point[i][2]),
                            calculate_distance(corner_point[i][2], corner_point[i][3]),
                            calculate_distance(corner_point[i][3], corner_point[i][0])
                        ]
                        size = np.mean(dists)
                        marker_sizes.append((self.ids[i], size))

                        # Check if marker size within tolerance
                        if not (self.EXPECTED_MARKER_SIZE - self.MARKER_SIZE_TOLERANCE
                                <= size <=
                                self.EXPECTED_MARKER_SIZE + self.MARKER_SIZE_TOLERANCE):
                            valid_sizes = False
                            print(f"[Warning] Marker ID {self.ids[i]} has invalid size: {size:.2f} mm")
                    except Exception as e:
                        print(f"Error calculating marker size for ID {self.ids[i]}: {e}")
                        valid_sizes = False
                        marker_sizes.append((self.ids[i], 0))
                else:
                    marker_sizes.append((self.ids[i], "not detected"))
            else:
                marker_sizes.append((self.ids[i], "not detected"))

        # Step 4: Return data only if detected markers have valid sizes
        if detected_count == 0:
            print(f"[Warning] No markers detected. Returning empty data.")
            empty_corners = np.zeros((len(self.ids), 4, 3))
            empty_centers = np.zeros((len(self.ids), 3))
            return empty_corners, empty_centers, None
        
        if not valid_sizes:
            print(f"[Warning] Invalid marker sizes detected: {marker_sizes}. Returning empty data.")
            empty_corners = np.zeros((len(self.ids), 4, 3))
            empty_centers = np.zeros((len(self.ids), 3))
            return empty_corners, empty_centers, None

        # Step 5: Return validated data
        print(f"[Success] Valid markers detected ({detected_count}/{len(self.ids)}): {marker_sizes}")
        return corner_point, center_point, frames


    
    # def send_data(self, return_median=False):  # Changed default to False
    #     Main method to get data - returns latest frame data by default
    #     Args:
    #         return_median: If True, returns median values from history. If False, returns latest frame data.
    
    #     if not self.running:
    #         self.start_threads()
    #         time.sleep(0.1)
        
    #     # Get latest frame data directly
    #     corner_point_matrix, center_points_matrix, frames = self.get_latest_data()
        
    #     if corner_point_matrix is not None:
    #         return corner_point_matrix, center_points_matrix, frames
        
    #     # Fallback to empty data if no processed data available
    #     empty_corners = np.zeros((len(self.ids), 4, 3))
    #     empty_centers = np.zeros((len(self.ids), 3))
    #     return empty_corners, empty_centers, None
    
    def __del__(self):
       '''Cleanup when object is destroyed'''
       self.stop_threads()

class data_1_main:
    def __init__(self):
        self.ids = [8, 5, 11, 13]
        self.StereoObj = SterioParameter("cal_10X7_A4_27.pkl")
        self.obj = ArUco()
        self.obj.set_tool_vector(self.ids)
        self.obj2 = CameraCapture_USB(resolution=(2560, 720))
        self.corner_points_history = deque(maxlen=7)
        self.center_points_history = deque(maxlen=7)
        
        # Threading setup for better performance
        self.frame_queue = Queue(maxsize=2)
        self.processed_data_queue = Queue(maxsize=7)
        self.capture_thread = None
        self.processing_thread = None
        self.running = False
        
        # Locks for thread safety
        self.history_lock = threading.Lock()
        self.camera_lock = threading.Lock()  # Lock for camera access
        
        # Error handling
        self.camera_initialized = True
        
        # Dynamic marker size calibration
        self.CALIBRATION_DURATION = 15.0  # seconds
        self.MARKER_SIZE_TOLERANCE = 1.5  # mm
        self.EXPECTED_MARKER_SIZE = None  # Will be set after calibration
        
        self.calibration_mode = True
        self.calibration_start_time = None
        self.calibration_samples = []  # Store all marker sizes during calibration
        self.calibration_lock = threading.Lock()

    def start_threads(self):
        """Start the capture and processing threads"""
        self.running = True
        self.calibration_start_time = time.time()
        
        # Start frame capture thread
        self.capture_thread = threading.Thread(target=self._frame_capture_thread, daemon=True)
        self.capture_thread.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_thread, daemon=True)
        self.processing_thread.start()
        
    def stop_threads(self):
        """Stop all threads gracefully"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
    
    def _frame_capture_thread(self):
        """Continuously capture frames in a separate thread"""
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while self.running:
            try:
                ret, left, right, frames = self.obj2.get_IR_FRAME_SET()
                if ret:
                    consecutive_errors = 0  # Reset error counter on success
                    frame_data = (left, right, frames)
                    
                    # If queue is full, remove old frame and add new one
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except Empty:
                            pass
                    
                    try:
                        self.frame_queue.put(frame_data, timeout=0.01)
                    except:
                        # Queue is full, skip this frame
                        continue
                else:
                    # Camera didn't return a frame, small delay
                    time.sleep(0.01)
                        
            except Exception as e:
                consecutive_errors += 1
                print(f"Error in frame capture thread: {e}")
                
                # If too many consecutive errors, increase delay
                if consecutive_errors > max_consecutive_errors:
                    print("Too many consecutive camera errors, using longer delay...")
                    time.sleep(0.1)
                else:
                    time.sleep(0.01)
                
                # Reset camera connection if too many errors
                if consecutive_errors > 20:
                    print("Attempting to reset camera connection...")
                    try:
                        # Reinitialize camera object
                        self.obj2 = CameraCapture_USB(CAMERA_CONFIG_OBJ=None)
                        consecutive_errors = 0
                    except Exception as reset_error:
                        print(f"Failed to reset camera: {reset_error}")
                        time.sleep(1.0)
    
    def _collect_calibration_sample(self, corner_point):
        """Collect marker size samples during calibration phase"""
        for i in range(len(self.ids)):
            # Check if marker is detected
            if len(corner_point) > i and corner_point[i] is not None and corner_point[i].size != 0:
                # Check if it's not just zeros
                if not np.allclose(corner_point[i], 0):
                    try:
                        # Calculate side lengths of each detected marker
                        dists = [
                            calculate_distance(corner_point[i][0], corner_point[i][1]),
                            calculate_distance(corner_point[i][1], corner_point[i][2]),
                            calculate_distance(corner_point[i][2], corner_point[i][3]),
                            calculate_distance(corner_point[i][3], corner_point[i][0])
                        ]
                        size = np.mean(dists)
                        
                        with self.calibration_lock:
                            self.calibration_samples.append(size)
                    except Exception as e:
                        print(f"Error collecting calibration sample for ID {self.ids[i]}: {e}")
    
    def _finalize_calibration(self):
        """Calculate the expected marker size from collected samples"""
        with self.calibration_lock:
            if len(self.calibration_samples) > 0:
                # Calculate average marker size
                self.EXPECTED_MARKER_SIZE = np.mean(self.calibration_samples)
                std_dev = np.std(self.calibration_samples)
                
                print(f"\n{'='*60}")
                print(f"CALIBRATION COMPLETE")
                print(f"{'='*60}")
                print(f"Samples collected: {len(self.calibration_samples)}")
                print(f"Average marker size: {self.EXPECTED_MARKER_SIZE:.2f} mm")
                print(f"Standard deviation: {std_dev:.2f} mm")
                print(f"Tolerance range: {self.EXPECTED_MARKER_SIZE - self.MARKER_SIZE_TOLERANCE:.2f} mm "
                      f"to {self.EXPECTED_MARKER_SIZE + self.MARKER_SIZE_TOLERANCE:.2f} mm")
                print(f"{'='*60}\n")
            else:
                # Fallback to default value
                self.EXPECTED_MARKER_SIZE = 50.0
                print(f"\n[Warning] No calibration samples collected. Using default size: {self.EXPECTED_MARKER_SIZE} mm\n")
            
            self.calibration_mode = False
    
    def _check_calibration_status(self):
        """Check if calibration period has elapsed"""
        if self.calibration_mode and self.calibration_start_time is not None:
            elapsed_time = time.time() - self.calibration_start_time
            
            # Show progress every 5 seconds
            if int(elapsed_time) % 5 == 0 and len(self.calibration_samples) > 0:
                with self.calibration_lock:
                    current_avg = np.mean(self.calibration_samples)
                    print(f"[Calibration] {elapsed_time:.0f}s / {self.CALIBRATION_DURATION:.0f}s - "
                          f"Samples: {len(self.calibration_samples)}, Current avg: {current_avg:.2f} mm")
            
            if elapsed_time >= self.CALIBRATION_DURATION:
                self._finalize_calibration()
    
    def _processing_thread(self):
        """Process frames in a separate thread"""
        while self.running:
            try:
                # Get frame from queue with timeout
                left, right, frames = self.frame_queue.get(timeout=1.0)
                
                # Process the frame
                corner_tool_ret, corner_point = self.obj.get_corner_world_points(self.StereoObj, left, right)
                center_tool_ret, center_points = self.obj.get_center_world_points(self.StereoObj, left, right)
                
                # Handle None values in corner_point
                corner_point_processed = []
                for cp in corner_point:
                    if cp is None or cp.size == 0:
                        cp = np.zeros((4, 3))
                    corner_point_processed.append(cp)
                
                # Convert corner_point to a single 3D NumPy array
                corner_point_matrix = np.array(corner_point_processed)
                
                # Collect calibration samples if in calibration mode
                if self.calibration_mode:
                    self._collect_calibration_sample(corner_point_matrix)
                    self._check_calibration_status()
                
                # Handle None values in center_points
                center_points_processed = []
                for cp in center_points:
                    if cp is None or cp.size == 0:
                        cp = np.zeros((1, 3))
                    center_points_processed.append(cp.flatten())
                
                # Convert center_points to a 2D NumPy array
                center_points_matrix = np.vstack(center_points_processed)
                
                # Thread-safe update of history
                with self.history_lock:
                    self.corner_points_history.append(corner_point_matrix)
                    self.center_points_history.append(center_points_matrix)
                
                # Put processed data in output queue
                processed_data = (corner_point_matrix, center_points_matrix, frames)
                
                if self.processed_data_queue.full():
                    try:
                        self.processed_data_queue.get_nowait()
                    except Empty:
                        pass
                
                try:
                    self.processed_data_queue.put(processed_data, timeout=0.01)
                except:
                    continue
                    
            except Empty:
                # No frame available, continue
                continue
            except Exception as e:
                print(f"Error in processing thread: {e}")
                continue
    
    def get_latest_data(self, timeout=1.0):
        """Get the latest processed data"""
        try:
            return self.processed_data_queue.get(timeout=timeout)
        except Empty:
            return None, None, None
    
    def get_median_data(self):
        """Get median values from history (thread-safe)"""
        with self.history_lock:
            if len(self.corner_points_history) >= 7:
                median_corner_points = np.mean(list(self.corner_points_history), axis=0)
                median_center_points = np.mean(list(self.center_points_history), axis=0)
                print("median values")
                return median_corner_points, median_center_points
            elif len(self.corner_points_history) > 0:
                # Return the latest data if we don't have enough history
                return self.corner_points_history[-1], self.center_points_history[-1]
            else:
                return None, None
    
    def is_calibrated(self):
        """Check if calibration is complete"""
        return not self.calibration_mode
    
    def send_data(self, return_median=True):
        """
        Fetches processed marker data and performs validation.
        Only returns data if marker sizes are valid.
        Returns empty arrays if validation fails.
        """
        # Start threads if not running
        if not self.running:
            self.start_threads()
            time.sleep(0.1)  # allow threads to warm up

        # If still in calibration mode, return empty data
        if self.calibration_mode:
            print(f"[Calibration Mode] Please wait... ({time.time() - self.calibration_start_time:.1f}s / {self.CALIBRATION_DURATION}s)")
            empty_corners = np.zeros((len(self.ids), 4, 3))
            empty_centers = np.zeros((len(self.ids), 3))
            return empty_corners, empty_centers, None

        # Step 1: Fetch data (either median or latest)
        if return_median:
            corner_point, center_point = self.get_median_data()
            _, _, frames = self.get_latest_data()
        else:
            corner_point, center_point, frames = self.get_latest_data()

        # Step 2: Handle missing data
        if corner_point is None or center_point is None:
            empty_corners = np.zeros((len(self.ids), 4, 3))
            empty_centers = np.zeros((len(self.ids), 3))
            return empty_corners, empty_centers, None

        # Step 3: Validate marker sizes (only for detected markers)
        valid_sizes = True
        marker_sizes = []
        detected_count = 0

        for i in range(len(self.ids)):
            # Check if marker is detected
            if len(corner_point) > i and corner_point[i] is not None and corner_point[i].size != 0:
                # Check if it's not just zeros
                if not np.allclose(corner_point[i], 0):
                    detected_count += 1
                    try:
                        # Calculate side lengths of each detected marker
                        dists = [
                            calculate_distance(corner_point[i][0], corner_point[i][1]),
                            calculate_distance(corner_point[i][1], corner_point[i][2]),
                            calculate_distance(corner_point[i][2], corner_point[i][3]),
                            calculate_distance(corner_point[i][3], corner_point[i][0])
                        ]
                        size = np.mean(dists)
                        marker_sizes.append((self.ids[i], size))

                        # Check if marker size within tolerance
                        if not (self.EXPECTED_MARKER_SIZE - self.MARKER_SIZE_TOLERANCE
                                <= size <=
                                self.EXPECTED_MARKER_SIZE + self.MARKER_SIZE_TOLERANCE):
                            valid_sizes = False
                            print(f"[Warning] Marker ID {self.ids[i]} has invalid size: {size:.2f} mm "
                                  f"(expected: {self.EXPECTED_MARKER_SIZE:.2f} ± {self.MARKER_SIZE_TOLERANCE} mm)")
                    except Exception as e:
                        print(f"Error calculating marker size for ID {self.ids[i]}: {e}")
                        valid_sizes = False
                        marker_sizes.append((self.ids[i], 0))
                else:
                    marker_sizes.append((self.ids[i], "not detected"))
            else:
                marker_sizes.append((self.ids[i], "not detected"))

        # Step 4: Return data only if detected markers have valid sizes
        if detected_count == 0:
            print(f"[Warning] No markers detected. Returning empty data.")
            empty_corners = np.zeros((len(self.ids), 4, 3))
            empty_centers = np.zeros((len(self.ids), 3))
            return empty_corners, empty_centers, None
        
        if not valid_sizes:
            print(f"[Warning] Invalid marker sizes detected: {marker_sizes}. Returning empty data.")
            empty_corners = np.zeros((len(self.ids), 4, 3))
            empty_centers = np.zeros((len(self.ids), 3))
            return empty_corners, empty_centers, None

        # Step 5: Return validated data
        print(f"[Success] Valid markers detected ({detected_count}/{len(self.ids)}): {marker_sizes}")
        return corner_point, center_point, frames
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop_threads()


class data_1:
    def __init__(self):
        self.ids = [8, 5, 11, 13]
        self.StereoObj = SterioParameter("cal_10X7_A4_27.pkl")
        self.obj = ArUco()
        self.obj.set_tool_vector(self.ids)
        self.obj2 = CameraCapture_USB(resolution=(2560, 720))
        self.corner_points_history = deque(maxlen=11)
        self.center_points_history = deque(maxlen=11)
        
        # Threading setup
        self.frame_queue = Queue(maxsize=2)
        self.processed_data_queue = Queue(maxsize=11)
        self.capture_thread = None
        self.processing_thread = None
        self.running = False
        
        # Locks for thread safety
        self.history_lock = threading.Lock()
        self.camera_lock = threading.Lock()
        
        # Error handling
        self.camera_initialized = True
        
        # Dynamic marker size calibration
        self.CALIBRATION_DURATION = 15.0  # seconds
        self.MARKER_SIZE_TOLERANCE = 1.5  # mm
        self.EXPECTED_MARKER_SIZE = None
        
        # NEW: Angle validation parameters
        self.ANGLE_TOLERANCE = 3.0  # degrees (±2°)
        self.ENABLE_ANGLE_VALIDATION = True  # Can be toggled
        
        self.calibration_mode = True
        self.calibration_start_time = None
        self.calibration_samples = []
        self.calibration_lock = threading.Lock()

    # ==================== ANGLE VALIDATION METHODS ====================
    
    @staticmethod
    def calculate_plane_normal(points):
        """
        Calculate normal vector of a plane defined by 3 or 4 points.
        Uses cross product of two vectors in the plane.
        
        Args:
            points: numpy array of shape (3 or 4, 3) - 3D coordinates
        
        Returns:
            numpy array of shape (3,) - unit normal vector
        """
        if len(points) < 3:
            return None
        
        # Use first 3 points to define plane
        p1, p2, p3 = points[0], points[1], points[2]
        
        # Create two vectors in the plane
        v1 = p2 - p1
        v2 = p3 - p1
        
        # Normal is cross product
        normal = np.cross(v1, v2)
        
        # Normalize to unit vector
        norm = np.linalg.norm(normal)
        if norm < 1e-6:  # Prevent division by zero
            return None
        
        return normal / norm
    
    @staticmethod
    def angle_between_normals(normal1, normal2):
        """
        Calculate angle between two normal vectors in degrees.
        Handles both 0° and 180° cases (parallel/antiparallel).
        
        Args:
            normal1: numpy array of shape (3,)
            normal2: numpy array of shape (3,)
        
        Returns:
            float: angle in degrees (0-180)
        """
        if normal1 is None or normal2 is None:
            return None
        
        # Normalize vectors
        n1 = normal1 / np.linalg.norm(normal1)
        n2 = normal2 / np.linalg.norm(normal2)
        
        # Calculate dot product (clipped to avoid numerical errors)
        dot_product = np.clip(np.dot(n1, n2), -1.0, 1.0)
        
        # Calculate angle in radians then convert to degrees
        angle_rad = np.arccos(abs(dot_product))  # abs() handles 0° and 180°
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def validate_marker_planarity(self, corner_points, marker_id):
        """
        Validate that marker corners are coplanar by checking angles
        between two planes formed by different point combinations.
        
        Strategy:
        - Plane 1: Points [0, 1, 2] (top-left, top-right, bottom-right)
        - Plane 2: Points [0, 2, 3] (top-left, bottom-right, bottom-left)
        
        If marker is flat, these planes should be parallel (angle ≈ 0° or 180°)
        
        Args:
            corner_points: numpy array of shape (4, 3) - marker corners
            marker_id: int - marker ID for logging
        
        Returns:
            tuple: (is_valid: bool, angle: float or None)
        """
        try:
            if corner_points.shape != (4, 3):
                return False, None
            
            # Check if all corners are non-zero (marker detected)
            if np.allclose(corner_points, 0):
                return False, None
            
            # Calculate two plane normals using different point combinations
            # Plane 1: top-left, top-right, bottom-right
            plane1_points = corner_points[[0, 1, 2]]
            normal1 = self.calculate_plane_normal(plane1_points)
            
            # Plane 2: top-left, bottom-right, bottom-left
            plane2_points = corner_points[[0, 2, 3]]
            normal2 = self.calculate_plane_normal(plane2_points)
            
            if normal1 is None or normal2 is None:
                print(f"[Angle Check] Marker {marker_id}: Failed to calculate normals")
                return False, None
            
            # Calculate angle between planes
            angle = self.angle_between_normals(normal1, normal2)
            
            if angle is None:
                return False, None
            
            # Check if angle is within tolerance (should be close to 0° or 180°)
            # Since we use abs() in angle calculation, we only check against 0°
            is_valid = angle <= self.ANGLE_TOLERANCE
            
            if not is_valid:
                print(f"[Angle Check] Marker {marker_id}: FAILED - Angle={angle:.2f}° "
                      f"(tolerance: ±{self.ANGLE_TOLERANCE}°)")
            else:
                print(f"[Angle Check] Marker {marker_id}: PASSED - Angle={angle:.2f}°")
            
            return is_valid, angle
            
        except Exception as e:
            print(f"[Angle Check] Error validating marker {marker_id}: {e}")
            return False, None
    
    def validate_marker_geometry(self, corner_points, marker_id):
        """
        Enhanced validation combining:
        1. Planarity check (angle between diagonal planes)
        2. Shape check (aspect ratio, orthogonality)
        
        Args:
            corner_points: numpy array of shape (4, 3)
            marker_id: int
        
        Returns:
            tuple: (is_valid: bool, details: dict)
        """
        details = {
            'planarity_valid': False,
            'planarity_angle': None,
            'aspect_ratio_valid': False,
            'aspect_ratio': None,
            'orthogonality_valid': False,
            'orthogonality_angle': None
        }
        
        try:
            # Check 1: Planarity
            planarity_valid, planarity_angle = self.validate_marker_planarity(corner_points, marker_id)
            details['planarity_valid'] = planarity_valid
            details['planarity_angle'] = planarity_angle
            
            if not planarity_valid:
                return False, details
            
            # Check 2: Aspect ratio (should be close to 1.0 for square markers)
            try:
                # Calculate side lengths
                side1 = np.linalg.norm(corner_points[1] - corner_points[0])  # top
                side2 = np.linalg.norm(corner_points[2] - corner_points[1])  # right
                side3 = np.linalg.norm(corner_points[3] - corner_points[2])  # bottom
                side4 = np.linalg.norm(corner_points[0] - corner_points[3])  # left
                
                # Calculate aspect ratio (horizontal vs vertical)
                horizontal = (side1 + side3) / 2
                vertical = (side2 + side4) / 2
                aspect_ratio = horizontal / vertical if vertical > 0 else 0
                
                details['aspect_ratio'] = aspect_ratio
                
                # Square markers should have aspect ratio close to 1.0
                # Allow ±20% deviation
                aspect_ratio_valid = 0.8 <= aspect_ratio <= 1.2
                details['aspect_ratio_valid'] = aspect_ratio_valid
                
                if not aspect_ratio_valid:
                    print(f"[Geometry Check] Marker {marker_id}: Invalid aspect ratio={aspect_ratio:.2f}")
                
            except Exception as e:
                print(f"[Geometry Check] Aspect ratio error for marker {marker_id}: {e}")
                details['aspect_ratio_valid'] = False
            
            # Check 3: Orthogonality (corners should form ~90° angles)
            try:
                # Check one corner (top-left)
                v1 = corner_points[1] - corner_points[0]  # to top-right
                v2 = corner_points[3] - corner_points[0]  # to bottom-left
                
                # Normalize
                v1 = v1 / np.linalg.norm(v1)
                v2 = v2 / np.linalg.norm(v2)
                
                # Calculate angle
                dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
                angle = np.degrees(np.arccos(abs(dot)))
                
                details['orthogonality_angle'] = angle
                
                # Should be close to 90° (±10°)
                orthogonality_valid = 80 <= angle <= 100
                details['orthogonality_valid'] = orthogonality_valid
                
                if not orthogonality_valid:
                    print(f"[Geometry Check] Marker {marker_id}: Non-orthogonal angle={angle:.2f}°")
                
            except Exception as e:
                print(f"[Geometry Check] Orthogonality error for marker {marker_id}: {e}")
                details['orthogonality_valid'] = False
            
            # Overall validity: planarity is REQUIRED, others are warnings
            is_valid = details['planarity_valid']
            
            return is_valid, details
            
        except Exception as e:
            print(f"[Geometry Check] Error for marker {marker_id}: {e}")
            return False, details

    # ==================== EXISTING METHODS (UPDATED) ====================
    
    def start_threads(self):
        """Start the capture and processing threads"""
        self.running = True
        #self.calibration_start_time = time.time()
        
        self.capture_thread = threading.Thread(target=self._frame_capture_thread, daemon=True)
        self.capture_thread.start()
        
        self.processing_thread = threading.Thread(target=self._processing_thread, daemon=True)
        self.processing_thread.start()
        
    def stop_threads(self):
        """Stop all threads gracefully"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
    
    def _frame_capture_thread(self):
        """Continuously capture frames in a separate thread"""
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while self.running:
            try:
                ret, left, right, frames = self.obj2.get_IR_FRAME_SET()
                if ret:
                    consecutive_errors = 0
                    frame_data = (left, right, frames)
                    
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except Empty:
                            pass
                    
                    try:
                        self.frame_queue.put(frame_data, timeout=0.01)
                    except:
                        continue
                else:
                    time.sleep(0.01)
                        
            except Exception as e:
                consecutive_errors += 1
                print(f"Error in frame capture thread: {e}")
                
                if consecutive_errors > max_consecutive_errors:
                    print("Too many consecutive camera errors, using longer delay...")
                    time.sleep(0.1)
                else:
                    time.sleep(0.01)
                
                if consecutive_errors > 20:
                    print("Attempting to reset camera connection...")
                    try:
                        self.obj2 = CameraCapture_USB(CAMERA_CONFIG_OBJ=None)
                        consecutive_errors = 0
                    except Exception as reset_error:
                        print(f"Failed to reset camera: {reset_error}")
                        time.sleep(1.0)
    
    def _collect_calibration_sample(self, corner_point):
        """Collect marker size samples during calibration phase"""
        for i in range(len(self.ids)):
            if len(corner_point) > i and corner_point[i] is not None and corner_point[i].size != 0:
                if not np.allclose(corner_point[i], 0):
                    if self.calibration_start_time is None:
                        self.calibration_start_time = time.time()
                        print(f"\n{'='*60}")
                        print(f"FIRST MARKER DETECTED - CALIBRATION STARTED")
                        print(f"{'='*60}\n")
                    try:
                        dists = [
                            calculate_distance(corner_point[i][0], corner_point[i][1]),
                            calculate_distance(corner_point[i][1], corner_point[i][2]),
                            calculate_distance(corner_point[i][2], corner_point[i][3]),
                            calculate_distance(corner_point[i][3], corner_point[i][0])
                        ]
                        size = np.mean(dists)
                        
                        with self.calibration_lock:
                            self.calibration_samples.append(size)
                    except Exception as e:
                        print(f"Error collecting calibration sample for ID {self.ids[i]}: {e}")
    
    def _finalize_calibration(self):
        """Calculate the expected marker size from collected samples"""
        with self.calibration_lock:
            if len(self.calibration_samples) > 0:
                self.EXPECTED_MARKER_SIZE = np.mean(self.calibration_samples)
                std_dev = np.std(self.calibration_samples)
                
                print(f"\n{'='*60}")
                print(f"CALIBRATION COMPLETE")
                print(f"{'='*60}")
                print(f"Samples collected: {len(self.calibration_samples)}")
                print(f"Average marker size: {self.EXPECTED_MARKER_SIZE:.2f} mm")
                print(f"Standard deviation: {std_dev:.2f} mm")
                print(f"Tolerance range: {self.EXPECTED_MARKER_SIZE - self.MARKER_SIZE_TOLERANCE:.2f} mm "
                      f"to {self.EXPECTED_MARKER_SIZE + self.MARKER_SIZE_TOLERANCE:.2f} mm")
                print(f"Angle validation: {'ENABLED' if self.ENABLE_ANGLE_VALIDATION else 'DISABLED'} "
                      f"(±{self.ANGLE_TOLERANCE}°)")
                print(f"{'='*60}\n")
            else:
                self.EXPECTED_MARKER_SIZE = 50.0
                print(f"\n[Warning] No calibration samples collected. Using default size: {self.EXPECTED_MARKER_SIZE} mm\n")
            
            self.calibration_mode = False
    
    def _check_calibration_status(self):
        """Check if calibration period has elapsed"""
        if self.calibration_mode and self.calibration_start_time is not None:
            elapsed_time = time.time() - self.calibration_start_time
            
            if int(elapsed_time) % 5 == 0 and len(self.calibration_samples) > 0:
                with self.calibration_lock:
                    current_avg = np.mean(self.calibration_samples)
                    print(f"[Calibration] {elapsed_time:.0f}s / {self.CALIBRATION_DURATION:.0f}s - "
                          f"Samples: {len(self.calibration_samples)}, Current avg: {current_avg:.2f} mm")
            
            if elapsed_time >= self.CALIBRATION_DURATION:
                self._finalize_calibration()
    
    def _processing_thread(self):
        """Process frames in a separate thread"""
        while self.running:
            try:
                left, right, frames = self.frame_queue.get(timeout=1.0)
                
                corner_tool_ret, corner_point = self.obj.get_corner_world_points(self.StereoObj, left, right)
                center_tool_ret, center_points = self.obj.get_center_world_points(self.StereoObj, left, right)
                
                corner_point_processed = []
                for cp in corner_point:
                    if cp is None or cp.size == 0:
                        cp = np.zeros((4, 3))
                    corner_point_processed.append(cp)
                
                corner_point_matrix = np.array(corner_point_processed)
                
                if self.calibration_mode:
                    self._collect_calibration_sample(corner_point_matrix)
                    self._check_calibration_status()
                
                center_points_processed = []
                for cp in center_points:
                    if cp is None or cp.size == 0:
                        cp = np.zeros((1, 3))
                    center_points_processed.append(cp.flatten())
                
                center_points_matrix = np.vstack(center_points_processed)
                
                with self.history_lock:
                    self.corner_points_history.append(corner_point_matrix)
                    self.center_points_history.append(center_points_matrix)
                
                processed_data = (corner_point_matrix, center_points_matrix, frames)
                
                if self.processed_data_queue.full():
                    try:
                        self.processed_data_queue.get_nowait()
                    except Empty:
                        pass
                
                try:
                    self.processed_data_queue.put(processed_data, timeout=0.01)
                except:
                    continue
                    
            except Empty:
                continue
            except Exception as e:
                print(f"Error in processing thread: {e}")
                continue
    
    def get_latest_data(self, timeout=1.0):
        """Get the latest processed data"""
        try:
            return self.processed_data_queue.get(timeout=timeout)
        except Empty:
            return None, None, None
    
    def get_median_data(self):
        """Get median values from history (thread-safe)"""
        with self.history_lock:
            if len(self.corner_points_history) >= 11:
                median_corner_points = np.mean(list(self.corner_points_history), axis=0)
                median_center_points = np.mean(list(self.center_points_history), axis=0)
                return median_corner_points, median_center_points
            elif len(self.corner_points_history) > 0:
                return self.corner_points_history[-1], self.center_points_history[-1]
            else:
                return None, None
    
    def is_calibrated(self):
        """Check if calibration is complete"""
        return not self.calibration_mode
    
    def send_data(self, return_median=True):
        """
        Fetches processed marker data and performs ENHANCED validation:
        1. Marker size validation
        2. Marker planarity validation (angle check)
        3. Geometry validation (aspect ratio, orthogonality)
        
        Returns empty arrays if validation fails.
        """
        # Start threads if not running
        if not self.running:
            self.start_threads()
            time.sleep(0.1)

        # If still in calibration mode, return empty data
        if self.calibration_mode:
            if self.calibration_start_time is None:
                print(f"[Calibration Mode] Waiting for first marker detection...")
            else:
                elapsed = time.time() - self.calibration_start_time
                print(f"[Calibration Mode] Please wait... ({elapsed:.1f}s / {self.CALIBRATION_DURATION}s)")
    
            empty_corners = np.zeros((len(self.ids), 4, 3))
            empty_centers = np.zeros((len(self.ids), 3))
            return empty_corners, empty_centers, None

        # Step 1: Fetch data
        if return_median:
            corner_point, center_point = self.get_median_data()
            _, _, frames = self.get_latest_data()
        else:
            corner_point, center_point, frames = self.get_latest_data()

        # Step 2: Handle missing data
        if corner_point is None or center_point is None:
            empty_corners = np.zeros((len(self.ids), 4, 3))
            empty_centers = np.zeros((len(self.ids), 3))
            return empty_corners, empty_centers, None

        # Step 3: Validate marker sizes AND angles
        valid_markers = True
        marker_validation_results = []
        detected_count = 0

        for i in range(len(self.ids)):
            marker_id = self.ids[i]
            
            # Check if marker is detected
            if len(corner_point) > i and corner_point[i] is not None and corner_point[i].size != 0:
                if not np.allclose(corner_point[i], 0):
                    detected_count += 1
                    
                    try:
                        # ===== SIZE VALIDATION =====
                        dists = [
                            calculate_distance(corner_point[i][0], corner_point[i][1]),
                            calculate_distance(corner_point[i][1], corner_point[i][2]),
                            calculate_distance(corner_point[i][2], corner_point[i][3]),
                            calculate_distance(corner_point[i][3], corner_point[i][0])
                        ]
                        size = np.mean(dists)
                        
                        size_valid = (self.EXPECTED_MARKER_SIZE - self.MARKER_SIZE_TOLERANCE
                                     <= size <=
                                     self.EXPECTED_MARKER_SIZE + self.MARKER_SIZE_TOLERANCE)
                        
                        if not size_valid:
                            valid_markers = False
                            print(f"[Size Check] Marker {marker_id}: FAILED - Size={size:.2f} mm "
                                  f"(expected: {self.EXPECTED_MARKER_SIZE:.2f} ± {self.MARKER_SIZE_TOLERANCE} mm)")
                        
                        # ===== ANGLE VALIDATION (NEW) =====
                        angle_valid = True
                        angle_details = None
                        
                        if self.ENABLE_ANGLE_VALIDATION:
                            angle_valid, angle_details = self.validate_marker_geometry(corner_point[i], marker_id)
                            
                            if not angle_valid:
                                valid_markers = False
                        
                        marker_validation_results.append({
                            'id': marker_id,
                            'size': size,
                            'size_valid': size_valid,
                            'angle_valid': angle_valid,
                            'angle_details': angle_details
                        })
                        
                    except Exception as e:
                        print(f"Error validating marker {marker_id}: {e}")
                        valid_markers = False
                        marker_validation_results.append({
                            'id': marker_id,
                            'size': 0,
                            'size_valid': False,
                            'angle_valid': False,
                            'angle_details': None
                        })
                else:
                    marker_validation_results.append({
                        'id': marker_id,
                        'detected': False
                    })
            else:
                marker_validation_results.append({
                    'id': marker_id,
                    'detected': False
                })

        # Step 4: Return data only if ALL detected markers are valid
        if detected_count == 0:
            print(f"[Warning] No markers detected. Returning empty data.")
            empty_corners = np.zeros((len(self.ids), 4, 3))
            empty_centers = np.zeros((len(self.ids), 3))
            return empty_corners, empty_centers, None
        
        if not valid_markers:
            print(f"[Warning] Invalid markers detected. Returning empty data.")
            print(f"[Validation Summary] {marker_validation_results}")
            empty_corners = np.zeros((len(self.ids), 4, 3))
            empty_centers = np.zeros((len(self.ids), 3))
            return empty_corners, empty_corners, None

        # Step 5: Return validated data
        print(f"[Success] All detected markers valid ({detected_count}/{len(self.ids)})")
        return corner_point, center_point, frames
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop_threads()
        
# def Detection(ids):
#     StereoObj = SterioParameter("cal_10X7_A4_1.pkl")
#     obj = ArUco()
#     obj.set_tool_vector(ids)

#     # img_path = config['PATHS']['ArucoImagePath']
#     # img_format = config['IMAGE_FORMAT']['ArucoImageFormat']

#     # # reading images from folder
#     # left_images =  glob.glob(img_path + 'LEFT/*' + img_format)
#     # right_images = glob.glob(img_path + 'RIGHT/*' + img_format)
#     # left_images.sort()
#     # right_images.sort()

#     dist =[]
#     angle = []
#     far = []
#     angle_with_camera = []
#     Aruco_corners_distances = []
#     angle_errors = []
#     data_records = []
#     actual_corner_distance = 60.0
#     actual_angle = 0.0

#     max_angle = None
#     max_angle_with_camera = None
#     max_distance = -float('inf')

#     obj2 = ComeraCapture_RealSense(CAMERA_CONFIG_OBJ = None)
#     #obj2 = CameraCapture_USB(CAMERA_CONFIG_OBJ=None)
    

#     while True:  
#         ret, left, right = obj2.get_IR_FRAME_SET() #getting images from camera
#         corner_tool_ret , corner_point = obj.get_corner_world_points(StereoObj , left , right) #getting corner points of aruco
#         print("corner_point : " , corner_point)
#         center_tool_ret , center_points = obj.get_center_world_points(StereoObj , left , right) #getting center points of aruco
#         if(corner_tool_ret == [True]*len(ids) and center_tool_ret == [True]*len(ids)): #if all aruco detected
#             f = np.linalg.norm((center_points[0][0] + center_points[1][0])/2) #distance between camera and center of two aruco
#             d = calculate_distance(point1=center_points[0][0] , point2 = center_points[1][0]) #distance between two aruco
#             #d = calculate_distance(point1=corner_point[0][1], point2=corner_point[0][2])
#             a = angle_between_two_aruco(corner_point[0] , corner_point[1]) #angle between two aruco
#             p = calculate_perpendicular_angle(corner_point[0]) #perpendicular vector of aruco
#             z = angle_between_two_vectors(center_points[0][0], p)
#             Aruco_corners_distance = calculate_distance(point1=corner_point[0][0], point2=corner_point[0][1])
#             Aruco_corners_distance = calculate_distance(point1=corner_point[0][1], point2=corner_point[0][2])
#             Aruco_corners_distance = calculate_distance(point1=corner_point[0][2], point2=corner_point[0][3])
#             Aruco_corners_distance = calculate_distance(point1=corner_point[0][3], point2=corner_point[0][0])
#             #print("Aruco_corners_distance : " , Aruco_corners_distance)
#             Aruco_corners_distances.append(Aruco_corners_distance)
             
#             angle_with_camera.append(z)
#             far.append(f)
#             dist.append(d)
#             angle.append(a)
    
#             # for angle error
#             if Aruco_corners_distance <59.0 or Aruco_corners_distance > 60.5:
#                 print("Aruco_corners_distance : ", Aruco_corners_distance)
#                 angle_error = abs(a - actual_angle)
#                 angle_errors.append(angle_error)
#                 #angle.remove(a)
#                 print("Angle error : ", angle_error)
#             else:
#                 angle_error = 0
            
#             data_records.append({
#                 "Aruco 1 Corners": corner_point[0],
#                 "Aruco 2 Corners": corner_point[1],
#                 "Distance Between Markers": d,
#                 "Angle": a,
#                 "Angle Error": angle_error
#             })

#             # Check if the current ArUco corners distance is the maximum
#             # if Aruco_corners_distance > max_distance:
#             #     max_distance = Aruco_corners_distance
#             #     max_angle = a
#             #     max_angle_with_camera = z

#             print("far : " , f)
#             print("Distance : " , d) 
#             print("angle : " , a)
#             print("angle with camera : " , z)

#             cv2.putText(left, f"Distance: {d:.0f} mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
#             cv2.putText(left, f"Angle: {a:.0f} degree", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
 
#         else:
#             print("Unable to detect")
        
        
#         #displaying scaled image
        
                                
#         # cv2.imshow('left',cv2.resize(left,(1920,1080)))
#         # cv2.imshow('right',cv2.resize(right,(1920,1080)))
#         #Realsense camera
#         cv2.imshow('left',cv2.resize(left,(1280,720)))
#         cv2.imshow('right',cv2.resize(right,(1280,720)))

#         #break while loop when keyboad interrupt
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         #time.sleep(1)
#         time.sleep(0.1)
    
   
#     # Calculate the error for corner distances
#     mean_corner_distance = np.mean(Aruco_corners_distances)
#     mean_angle = np.mean(angle)
#     error = mean_corner_distance - actual_corner_distance
#     sorted_distances = sorted(Aruco_corners_distances)
#     median_corner_distance = np.median(sorted_distances)

    
#     # angle error mean
#     print("angle_errors : ", angle_errors)
#     mean_angle_error = np.mean(angle_errors) if angle_errors else 0
#     print("mean angle error : ", mean_angle_error)
    
#     # print("Aruco_corners_distances max : ", max_distance)
#     # print("Angle at max distance: ", max_angle)
#     # print("Angle with camera at max distance: ", max_angle_with_camera)
#     # print("Aruco_corners_distances max : " , max(Aruco_corners_distances))
#     print("mean corner distance : ", mean_corner_distance)
#     print("mean angle : ", mean_angle)
#     print("Distance error : ", error)
#     print("Aruco_corners_distances median : ", median_corner_distance)

#     with open('vector_4_cm.pkl' , 'wb') as fp:
#         pickle.dump([far,dist,angle,angle_with_camera],fp)
    
#     # Write data to CSV
#     with open('aruco_data.csv', mode='w', newline='') as file:
#         writer = csv.DictWriter(file, fieldnames=["Aruco 1 Corners", "Aruco 2 Corners", "Distance Between Markers", 
#                                                    "Angle", "Angle Error"])
#         writer.writeheader()
#         writer.writerows(data_records)

#     print("CSV file 'aruco_data.csv' has been created.")
    
#     return dist , far , angle


def plot(dist,far,angle):
    plt.subplot(1,2,1)
    plt.scatter(far,angle)
    plt.xlabel("Far")
    plt.ylabel("Angle")
    plt.subplot(1,2,2)
    plt.scatter(far, dist)
    plt.xlabel("Far")
    plt.ylabel("Distance")
    plt.show()

#if __name__ == '__main__':

    # Calibrate Camera
    #CalibrateCamera()

    # Detecting Aruco and calculating distance and angle

    #ids = [11,13]  # Change accordingly 
    #dist,far,angle =  Detection(ids)
    #ids = [8,5,11,13,6]
    #send_data()
    # plotting distance and angle

    #plot(dist,far,angle)
    
