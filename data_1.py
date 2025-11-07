import numpy as np
from collections import deque
import threading
import time

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
        self.ANGLE_TOLERANCE = 2.0  # degrees (±2°)
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
        self.calibration_start_time = time.time()
        
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
            print(f"[Calibration Mode] Please wait... ({time.time() - self.calibration_start_time:.1f}s / {self.CALIBRATION_DURATION}s)")
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