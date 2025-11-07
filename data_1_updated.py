import numpy as np
from collections import deque
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d
import threading

class OptimizedData_1:
    """
    Enhanced data class with multiple noise reduction strategies
    for improved angle validation accuracy.
    """
    
    def __init__(self):
        self.ids = [8, 5, 11, 13]
        self.StereoObj = SterioParameter("cal_10X7_A4_27.pkl")
        self.obj = ArUco()
        self.obj.set_tool_vector(self.ids)
        self.obj2 = CameraCapture_USB(resolution=(2560, 720))
        
        # ========== NOISE REDUCTION CONFIGURATION ==========
        
        # Strategy 1: Multi-level history with different purposes
        self.corner_points_raw_history = deque(maxlen=30)      # Raw data (longer)
        self.corner_points_filtered_history = deque(maxlen=11) # Filtered data (for median)
        self.center_points_history = deque(maxlen=11)
        
        # Strategy 2: Temporal consistency tracking
        self.angle_validation_history = {}  # Track angle validity over time
        for marker_id in self.ids:
            self.angle_validation_history[marker_id] = {
                'angles': deque(maxlen=10),
                'valid_count': 0,
                'total_count': 0
            }
        
        # Strategy 3: Adaptive filtering parameters
        self.NOISE_REDUCTION_MODE = "adaptive"  # Options: "off", "basic", "aggressive", "adaptive"
        self.TEMPORAL_WINDOW = 5                # Frames to consider for temporal filtering
        self.SPATIAL_SMOOTHING_SIGMA = 0.5      # Gaussian smoothing strength
        self.OUTLIER_Z_THRESHOLD = 2.5          # Z-score for outlier detection
        
        # Strategy 4: Corner stability tracking
        self.corner_stability_scores = {}
        for marker_id in self.ids:
            self.corner_stability_scores[marker_id] = deque(maxlen=10)
        
        # Threading and locks
        self.frame_queue = Queue(maxsize=2)
        self.processed_data_queue = Queue(maxsize=11)
        self.capture_thread = None
        self.processing_thread = None
        self.running = False
        self.history_lock = threading.Lock()
        self.camera_lock = threading.Lock()
        
        # Validation parameters
        self.CALIBRATION_DURATION = 15.0
        self.MARKER_SIZE_TOLERANCE = 1.5
        self.EXPECTED_MARKER_SIZE = None
        self.ANGLE_TOLERANCE = 2.0
        self.ENABLE_ANGLE_VALIDATION = True
        
        # Adaptive angle tolerance (NEW)
        self.USE_ADAPTIVE_TOLERANCE = True
        self.MIN_ANGLE_TOLERANCE = 1.5  # Strictest
        self.MAX_ANGLE_TOLERANCE = 3.0  # Most lenient
        
        self.calibration_mode = True
        self.calibration_start_time = None
        self.calibration_samples = []
        self.calibration_lock = threading.Lock()
    
    # ========== NOISE REDUCTION METHODS ==========
    
    def apply_temporal_median_filter(self, corner_points_sequence):
        """
        Apply median filter across time for each corner point.
        Removes temporal spikes while preserving true motion.
        
        Args:
            corner_points_sequence: List of corner_point arrays (T, N, 4, 3)
        
        Returns:
            numpy array: Temporally filtered corner points (N, 4, 3)
        """
        if len(corner_points_sequence) < 3:
            return corner_points_sequence[-1] if corner_points_sequence else None
        
        # Stack sequence along time axis
        sequence_array = np.array(list(corner_points_sequence))  # (T, N, 4, 3)
        
        # Apply median along time axis for each coordinate
        filtered = np.median(sequence_array, axis=0)  # (N, 4, 3)
        
        return filtered
    
    def apply_gaussian_smoothing(self, corner_points_sequence, sigma=0.5):
        """
        Apply Gaussian temporal smoothing.
        Smoother than median, good for continuous motion.
        
        Args:
            corner_points_sequence: List of corner arrays
            sigma: Standard deviation for Gaussian kernel
        
        Returns:
            numpy array: Smoothed corner points
        """
        if len(corner_points_sequence) < 3:
            return corner_points_sequence[-1] if corner_points_sequence else None
        
        sequence_array = np.array(list(corner_points_sequence))  # (T, N, 4, 3)
        
        # Apply Gaussian filter along time axis
        smoothed = gaussian_filter1d(sequence_array, sigma=sigma, axis=0)
        
        # Return the center frame (most weighted)
        center_idx = len(sequence_array) // 2
        return smoothed[center_idx]
    
    def detect_and_remove_outliers(self, corner_points_sequence, z_threshold=2.5):
        """
        Detect and remove outlier frames using Z-score method.
        
        Args:
            corner_points_sequence: List of corner arrays
            z_threshold: Z-score threshold for outlier detection
        
        Returns:
            list: Cleaned sequence with outliers removed
        """
        if len(corner_points_sequence) < 5:
            return corner_points_sequence
        
        sequence_array = np.array(list(corner_points_sequence))
        
        # Calculate mean and std for each coordinate
        mean = np.mean(sequence_array, axis=0)
        std = np.std(sequence_array, axis=0)
        
        # Calculate Z-scores
        z_scores = np.abs((sequence_array - mean) / (std + 1e-6))
        
        # Keep only frames where all coordinates are within threshold
        max_z_per_frame = np.max(z_scores, axis=(1, 2, 3))
        valid_mask = max_z_per_frame < z_threshold
        
        # Filter sequence
        cleaned_sequence = [corner_points_sequence[i] for i in range(len(valid_mask)) if valid_mask[i]]
        
        if len(cleaned_sequence) == 0:
            return corner_points_sequence  # Return original if all flagged as outliers
        
        return cleaned_sequence
    
    def calculate_corner_stability(self, corner_points_sequence):
        """
        Calculate stability score for corner points.
        Higher score = more stable (less noise).
        
        Args:
            corner_points_sequence: Recent corner point history
        
        Returns:
            dict: Stability scores per marker {marker_id: score}
        """
        if len(corner_points_sequence) < 3:
            return {mid: 0.0 for mid in self.ids}
        
        sequence_array = np.array(list(corner_points_sequence))
        
        stability_scores = {}
        for i, marker_id in enumerate(self.ids):
            # Calculate variance across time for this marker
            marker_sequence = sequence_array[:, i, :, :]  # (T, 4, 3)
            
            # Check if marker is detected (not all zeros)
            if np.allclose(marker_sequence[-1], 0):
                stability_scores[marker_id] = 0.0
                continue
            
            # Calculate standard deviation across time
            temporal_std = np.std(marker_sequence, axis=0)  # (4, 3)
            
            # Lower std = higher stability
            avg_std = np.mean(temporal_std)
            
            # Convert to 0-1 score (inverse of std)
            # Stable corners have std < 1mm, unstable > 5mm
            stability_score = 1.0 / (1.0 + avg_std)
            
            stability_scores[marker_id] = stability_score
        
        return stability_scores
    
    def adaptive_angle_tolerance(self, marker_id, stability_score):
        """
        Adjust angle tolerance based on corner stability.
        More stable = stricter tolerance, less stable = more lenient.
        
        Args:
            marker_id: Marker identifier
            stability_score: Stability score (0-1)
        
        Returns:
            float: Adjusted angle tolerance in degrees
        """
        if not self.USE_ADAPTIVE_TOLERANCE:
            return self.ANGLE_TOLERANCE
        
        # High stability (0.8-1.0) → strict tolerance (1.5°)
        # Low stability (0-0.2) → lenient tolerance (3.0°)
        tolerance = self.MAX_ANGLE_TOLERANCE - (stability_score * (self.MAX_ANGLE_TOLERANCE - self.MIN_ANGLE_TOLERANCE))
        
        return tolerance
    
    def apply_kalman_smoothing(self, corner_points_sequence):
        """
        Simple Kalman-like filtering for corner points.
        Predicts next position and corrects with measurement.
        
        Args:
            corner_points_sequence: Recent history
        
        Returns:
            numpy array: Kalman-filtered corner points
        """
        if len(corner_points_sequence) < 2:
            return corner_points_sequence[-1] if corner_points_sequence else None
        
        # Simple implementation: weighted average favoring recent data
        sequence_array = np.array(list(corner_points_sequence))
        
        # Create exponentially decaying weights (more recent = higher weight)
        weights = np.exp(np.linspace(-1, 0, len(sequence_array)))
        weights = weights / np.sum(weights)
        
        # Apply weighted average
        filtered = np.average(sequence_array, axis=0, weights=weights)
        
        return filtered
    
    def get_filtered_corner_points(self, raw_corner_points):
        """
        Master function to apply selected noise reduction strategy.
        
        Args:
            raw_corner_points: Current raw corner point data
        
        Returns:
            numpy array: Filtered corner points
        """
        # Add to raw history
        with self.history_lock:
            self.corner_points_raw_history.append(raw_corner_points)
        
        # Choose filtering strategy
        if self.NOISE_REDUCTION_MODE == "off":
            return raw_corner_points
        
        elif self.NOISE_REDUCTION_MODE == "basic":
            # Simple temporal median
            if len(self.corner_points_raw_history) >= 5:
                recent = list(self.corner_points_raw_history)[-5:]
                return self.apply_temporal_median_filter(recent)
            return raw_corner_points
        
        elif self.NOISE_REDUCTION_MODE == "aggressive":
            # Outlier removal + Gaussian smoothing
            if len(self.corner_points_raw_history) >= 7:
                recent = list(self.corner_points_raw_history)[-7:]
                cleaned = self.detect_and_remove_outliers(recent, z_threshold=2.0)
                return self.apply_gaussian_smoothing(cleaned, sigma=0.7)
            return raw_corner_points
        
        elif self.NOISE_REDUCTION_MODE == "adaptive":
            # Adaptive: adjust based on stability
            if len(self.corner_points_raw_history) >= 7:
                recent = list(self.corner_points_raw_history)[-7:]
                
                # Calculate stability
                stability = self.calculate_corner_stability(recent)
                
                # Store stability history
                for marker_id in self.ids:
                    self.corner_stability_scores[marker_id].append(stability[marker_id])
                
                # If high stability, use light filtering
                avg_stability = np.mean(list(stability.values()))
                
                if avg_stability > 0.7:
                    # High stability: light median filter
                    return self.apply_temporal_median_filter(recent[-3:])
                elif avg_stability > 0.4:
                    # Medium stability: Gaussian smoothing
                    return self.apply_gaussian_smoothing(recent, sigma=0.5)
                else:
                    # Low stability: aggressive outlier removal + smoothing
                    cleaned = self.detect_and_remove_outliers(recent, z_threshold=2.5)
                    return self.apply_gaussian_smoothing(cleaned, sigma=0.8)
            
            return raw_corner_points
        
        return raw_corner_points
    
    # ========== ENHANCED ANGLE VALIDATION ==========
    
    @staticmethod
    def calculate_plane_normal(points):
        """Calculate normal vector of a plane defined by 3 points."""
        if len(points) < 3:
            return None
        
        p1, p2, p3 = points[0], points[1], points[2]
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            return None
        
        return normal / norm
    
    @staticmethod
    def angle_between_normals(normal1, normal2):
        """Calculate angle between two normal vectors."""
        if normal1 is None or normal2 is None:
            return None
        
        n1 = normal1 / np.linalg.norm(normal1)
        n2 = normal2 / np.linalg.norm(normal2)
        
        dot_product = np.clip(np.dot(n1, n2), -1.0, 1.0)
        angle_rad = np.arccos(abs(dot_product))
        
        return np.degrees(angle_rad)
    
    def validate_marker_planarity_with_history(self, corner_points, marker_id):
        """
        Enhanced planarity validation with temporal consistency check.
        
        Args:
            corner_points: Current corner points (4, 3)
            marker_id: Marker identifier
        
        Returns:
            tuple: (is_valid: bool, angle: float, confidence: float)
        """
        try:
            if corner_points.shape != (4, 3):
                return False, None, 0.0
            
            if np.allclose(corner_points, 0):
                return False, None, 0.0
            
            # Calculate current angle
            plane1_points = corner_points[[0, 1, 2]]
            plane2_points = corner_points[[0, 2, 3]]
            normal1 = self.calculate_plane_normal(plane1_points)
            normal2 = self.calculate_plane_normal(plane2_points)
            
            if normal1 is None or normal2 is None:
                return False, None, 0.0
            
            current_angle = self.angle_between_normals(normal1, normal2)
            
            if current_angle is None:
                return False, None, 0.0
            
            # Store in history
            self.angle_validation_history[marker_id]['angles'].append(current_angle)
            self.angle_validation_history[marker_id]['total_count'] += 1
            
            # Get adaptive tolerance based on stability
            marker_idx = self.ids.index(marker_id)
            if len(self.corner_stability_scores[marker_id]) > 0:
                stability = np.mean(list(self.corner_stability_scores[marker_id]))
            else:
                stability = 0.5
            
            tolerance = self.adaptive_angle_tolerance(marker_id, stability)
            
            # Check temporal consistency
            recent_angles = list(self.angle_validation_history[marker_id]['angles'])
            
            if len(recent_angles) >= 3:
                # Calculate consistency score (low variance = high consistency)
                angle_variance = np.var(recent_angles[-5:])
                consistency = 1.0 / (1.0 + angle_variance)
            else:
                consistency = 0.5
            
            # Validate with adaptive tolerance
            is_valid = current_angle <= tolerance
            
            if is_valid:
                self.angle_validation_history[marker_id]['valid_count'] += 1
            
            # Calculate confidence based on:
            # 1. How close to 0° (ideal)
            # 2. Temporal consistency
            # 3. Historical success rate
            angle_quality = 1.0 - (current_angle / tolerance)
            
            total = self.angle_validation_history[marker_id]['total_count']
            valid = self.angle_validation_history[marker_id]['valid_count']
            success_rate = valid / total if total > 0 else 0.5
            
            confidence = (angle_quality * 0.5 + consistency * 0.3 + success_rate * 0.2)
            confidence = np.clip(confidence, 0.0, 1.0)
            
            if not is_valid:
                print(f"[Angle] Marker {marker_id}: FAIL - {current_angle:.2f}° > {tolerance:.2f}° "
                      f"(stability={stability:.2f}, confidence={confidence:.2f})")
            else:
                print(f"[Angle] Marker {marker_id}: PASS - {current_angle:.2f}° "
                      f"(tolerance={tolerance:.2f}°, confidence={confidence:.2f})")
            
            return is_valid, current_angle, confidence
            
        except Exception as e:
            print(f"[Angle Check] Error for marker {marker_id}: {e}")
            return False, None, 0.0
    
    # ========== MODIFIED PROCESSING THREAD ==========
    
    def _processing_thread(self):
        """Enhanced processing thread with noise reduction."""
        while self.running:
            try:
                left, right, frames = self.frame_queue.get(timeout=1.0)
                
                # Get raw marker data
                corner_tool_ret, corner_point = self.obj.get_corner_world_points(self.StereoObj, left, right)
                center_tool_ret, center_points = self.obj.get_center_world_points(self.StereoObj, left, right)
                
                # Process corners
                corner_point_processed = []
                for cp in corner_point:
                    if cp is None or cp.size == 0:
                        cp = np.zeros((4, 3))
                    corner_point_processed.append(cp)
                
                corner_point_matrix = np.array(corner_point_processed)
                
                # === APPLY NOISE REDUCTION ===
                filtered_corner_points = self.get_filtered_corner_points(corner_point_matrix)
                
                # Calibration
                if self.calibration_mode:
                    self._collect_calibration_sample(filtered_corner_points)
                    self._check_calibration_status()
                
                # Process centers
                center_points_processed = []
                for cp in center_points:
                    if cp is None or cp.size == 0:
                        cp = np.zeros((1, 3))
                    center_points_processed.append(cp.flatten())
                
                center_points_matrix = np.vstack(center_points_processed)
                
                # Update history with FILTERED data
                with self.history_lock:
                    self.corner_points_filtered_history.append(filtered_corner_points)
                    self.center_points_history.append(center_points_matrix)
                
                # Queue processed data
                processed_data = (filtered_corner_points, center_points_matrix, frames)
                
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
    
    def get_median_data(self):
        """Get median from filtered history."""
        with self.history_lock:
            if len(self.corner_points_filtered_history) >= 11:
                median_corner_points = np.median(list(self.corner_points_filtered_history), axis=0)
                median_center_points = np.median(list(self.center_points_history), axis=0)
                return median_corner_points, median_center_points
            elif len(self.corner_points_filtered_history) > 0:
                return self.corner_points_filtered_history[-1], self.center_points_history[-1]
            else:
                return None, None
    
    def send_data(self, return_median=True):
        """
        Enhanced send_data with comprehensive noise reduction and validation.
        """
        # [Same structure as before, but using filtered data and enhanced validation]
        
        if not self.running:
            self.start_threads()
            time.sleep(0.1)

        if self.calibration_mode:
            print(f"[Calibration Mode] Please wait... ({time.time() - self.calibration_start_time:.1f}s / {self.CALIBRATION_DURATION}s)")
            empty_corners = np.zeros((len(self.ids), 4, 3))
            empty_centers = np.zeros((len(self.ids), 3))
            return empty_corners, empty_centers, None

        # Fetch FILTERED data
        if return_median:
            corner_point, center_point = self.get_median_data()
            _, _, frames = self.get_latest_data()
        else:
            corner_point, center_point, frames = self.get_latest_data()

        if corner_point is None or center_point is None:
            empty_corners = np.zeros((len(self.ids), 4, 3))
            empty_centers = np.zeros((len(self.ids), 3))
            return empty_corners, empty_centers, None

        # Enhanced validation with history and confidence
        valid_markers = True
        marker_results = []
        detected_count = 0

        for i in range(len(self.ids)):
            marker_id = self.ids[i]
            
            if len(corner_point) > i and corner_point[i] is not None and corner_point[i].size != 0:
                if not np.allclose(corner_point[i], 0):
                    detected_count += 1
                    
                    try:
                        # Size validation
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
                        
                        # Enhanced angle validation with history
                        angle_valid = True
                        angle_value = None
                        confidence = 0.0
                        
                        if self.ENABLE_ANGLE_VALIDATION:
                            angle_valid, angle_value, confidence = self.validate_marker_planarity_with_history(
                                corner_point[i], marker_id
                            )
                        
                        # Overall validation: both size and angle must pass
                        is_valid = size_valid and angle_valid
                        
                        if not is_valid:
                            valid_markers = False
                        
                        marker_results.append({
                            'id': marker_id,
                            'size': size,
                            'size_valid': size_valid,
                            'angle_valid': angle_valid,
                            'angle': angle_value,
                            'confidence': confidence
                        })
                        
                    except Exception as e:
                        print(f"Error validating marker {marker_id}: {e}")
                        valid_markers = False

        if detected_count == 0:
            print(f"[Warning] No markers detected.")
            empty_corners = np.zeros((len(self.ids), 4, 3))
            empty_centers = np.zeros((len(self.ids), 3))
            return empty_corners, empty_centers, None
        
        if not valid_markers:
            print(f"[Warning] Invalid markers detected.")
            print(f"[Results] {marker_results}")
            empty_corners = np.zeros((len(self.ids), 4, 3))
            empty_centers = np.zeros((len(self.ids), 3))
            return empty_corners, empty_centers, None

        print(f"[Success] All detected markers valid ({detected_count}/{len(self.ids)})")
        return corner_point, center_point, frames