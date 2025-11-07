# flask_server.py - Fixed version without circular imports
from flask import Flask, render_template, jsonify, request, Response
from flask_cors import CORS
import threading
import time
import cv2
import numpy as np
import queue

app = Flask(__name__)
CORS(app)

class AppState:
    def __init__(self):
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.frame_queue = queue.Queue(maxsize=2)
        self.selected_object = None
        self.ct_side = None
        self.data_ready = False
        self.results = {
            'var_angle': 0,
            'var_flex': 0,
            'tmc_distance': 0,
            'tlc_distance': 0,
            'var_angle_name': '',
            'var_flex_name': ''
        }
        self.points_captured = {
            'femur': 0,
            'tibia': 0
        }
        self.is_processing = False
        self.messages = []
        self.capture_triggered = False
        self.reset_triggered = False
        self.frame_count = 0
        self.current_point_message = "-"
        
        # FPS tracking
        self.fps = 0
        self.fps_frame_count = 0
        self.fps_last_time = time.time()
        
        # Marker validation status
        self.marker_status = {
            'calibrated': False,
            'markers': {
                8: {'name': 'Pointer', 'valid': False, 'size': 0, 'detected': False},
                5: {'name': 'Verification', 'valid': False, 'size': 0, 'detected': False},
                11: {'name': 'Tibia', 'valid': False, 'size': 0, 'detected': False},
                13: {'name': 'Femur', 'valid': False, 'size': 0, 'detected': False}
            }
        }

app_state = AppState()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    try:
        return jsonify({
            'selected_object': app_state.selected_object if app_state.selected_object else 'tibia',
            'ct_side': app_state.ct_side if app_state.ct_side else None,
            'is_processing': app_state.is_processing,
            'points_captured': app_state.points_captured,
            'data_ready': app_state.data_ready,
            'fps': getattr(app_state, 'fps', 0),
            'total_frames': app_state.frame_count,
            'current_point': getattr(app_state, 'current_point_message', "-"),
            'marker_status': app_state.marker_status
        })
    except Exception as e:
        print(f"[Flask] Error in get_status: {e}")
        # Return minimal valid response on error
        return jsonify({
            'selected_object': 'tibia',
            'ct_side': None,
            'is_processing': False,
            'points_captured': {'femur': 0, 'tibia': 0},
            'data_ready': False,
            'fps': 0,
            'total_frames': 0,
            'current_point': '-',
            'marker_status': {
                'calibrated': False,
                'markers': {
                    8: {'name': 'Pointer', 'valid': False, 'size': 0, 'detected': False},
                    5: {'name': 'Verification', 'valid': False, 'size': 0, 'detected': False},
                    11: {'name': 'Tibia', 'valid': False, 'size': 0, 'detected': False},
                    13: {'name': 'Femur', 'valid': False, 'size': 0, 'detected': False}
                }
            }
        })

@app.route('/video_feed')
def video_feed():
    """MJPEG streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    """Generator for MJPEG frames"""
    while True:
        try:
            if not app_state.frame_queue.empty():
                frame = app_state.frame_queue.get_nowait()
                
                if frame is not None and frame.size > 0:
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                    success, buffer = cv2.imencode('.jpg', frame, encode_param)
                    
                    if success:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + 
                               buffer.tobytes() + b'\r\n')
            
            time.sleep(0.03)
            
        except queue.Empty:
            time.sleep(0.01)
        except Exception as e:
            print(f"[MJPEG] Error: {e}")
            time.sleep(0.1)

@app.route('/api/frame', methods=['GET'])
def get_frame():
    """Legacy base64 endpoint"""
    with app_state.frame_lock:
        if app_state.current_frame is not None and app_state.current_frame.size > 0:
            try:
                height, width = app_state.current_frame.shape[:2]
                if width > 640:
                    scale = 640 / width
                    new_width = 640
                    new_height = int(height * scale)
                    frame_resized = cv2.resize(app_state.current_frame, (new_width, new_height))
                else:
                    frame_resized = app_state.current_frame
                
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                success, buffer = cv2.imencode('.jpg', frame_resized, encode_param)
                
                if success:
                    import base64
                    frame_b64 = base64.b64encode(buffer).decode('utf-8')
                    return jsonify({'frame': frame_b64})
                else:
                    return jsonify({'error': 'Encoding failed'}), 500
            except Exception as e:
                print(f"[Frame Encode Error]: {e}")
                return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'No frame available'}), 400

@app.route('/api/results', methods=['GET'])
def get_results():
    try:
        return jsonify(app_state.results)
    except Exception as e:
        print(f"[Flask] Error in get_results: {e}")
        return jsonify({
            'var_angle': 0,
            'var_flex': 0,
            'tmc_distance': 0,
            'tlc_distance': 0,
            'var_angle_name': '',
            'var_flex_name': ''
        })

@app.route('/api/command', methods=['POST'])
def send_command():
    try:
        data = request.json
        command = data.get('command')
        
        if command == 'select_femur':
            app_state.selected_object = "femur"
            app_state.messages.append("Femur selected")
        
        elif command == 'select_tibia':
            app_state.selected_object = "tibia"
            app_state.messages.append("Tibia selected")
            # Don't import from main_keyboard here - causes circular import!
            # The main loop will handle updating the point message
            from main_keyboard import update_next_tibia_point
            update_next_tibia_point()
        
        elif command == 'set_ct_side':
            app_state.ct_side = data.get('side')
            app_state.messages.append(f"CT Side: {app_state.ct_side}")
        
        elif command == 'capture_point':
            app_state.capture_triggered = True
            app_state.is_processing = True
            app_state.messages.append("Capture initiated")
        
        elif command == 'reset':
            app_state.reset_triggered = True
            obj = app_state.selected_object
            if obj:
                app_state.points_captured[obj] = 0
            app_state.messages.append(f"Reset {obj}")
        
        elif command == 'analyze':
            app_state.data_ready = True
            app_state.messages.append("Analysis started")
        
        return jsonify({'status': 'Command received'})
    
    except Exception as e:
        print(f"[Flask] Error in send_command: {e}")
        return jsonify({'status': 'Error', 'message': str(e)}), 500

@app.route('/api/messages', methods=['GET'])
def get_messages():
    try:
        messages = app_state.messages[-10:]
        app_state.messages = []
        return jsonify({'messages': messages})
    except Exception as e:
        print(f"[Flask] Error in get_messages: {e}")
        return jsonify({'messages': []})

# ==================== HELPER FUNCTIONS ====================
def update_frame(frame):
    """Update current frame with FPS tracking"""
    if frame is None or frame.size == 0:
        return
    
    if not hasattr(app_state, 'fps_last_time'):
        app_state.fps_last_time = time.time()
        app_state.fps_frame_count = 0
        app_state.fps = 0
    
    with app_state.frame_lock:
        app_state.current_frame = frame.copy()
    
    try:
        if app_state.frame_queue.full():
            app_state.frame_queue.get_nowait()
        app_state.frame_queue.put_nowait(frame.copy())
        
        app_state.frame_count += 1
        app_state.fps_frame_count += 1
        current_time = time.time()
        elapsed = current_time - app_state.fps_last_time
        
        if elapsed >= 1.0:
            app_state.fps = round(app_state.fps_frame_count / elapsed, 1)
            app_state.fps_frame_count = 0
            app_state.fps_last_time = current_time
            # Only print every 5 seconds to reduce console spam
            if app_state.frame_count % 150 == 0:
                print(f"[Flask] FPS: {app_state.fps} | Total frames: {app_state.frame_count}")
        
    except queue.Full:
        pass
    except Exception as e:
        print(f"[Flask] Queue error: {e}")

def update_results(var_angle, var_flex, tmc_distance, tlc_distance, 
                   var_angle_name, var_flex_name):
    """Update analysis results"""
    try:
        app_state.results = {
            'var_angle': float(var_angle),
            'var_flex': float(var_flex),
            'tmc_distance': float(tmc_distance),
            'tlc_distance': float(tlc_distance),
            'var_angle_name': var_angle_name,
            'var_flex_name': var_flex_name,
            'timestamp': time.time()
        }
        app_state.data_ready = True
    except Exception as e:
        print(f"[Flask] Error in update_results: {e}")

def get_command_state():
    """Get current state from web"""
    try:
        return {
            'selected_object': app_state.selected_object,
            'ct_side': app_state.ct_side,
            'capture_point': app_state.capture_triggered,
            'reset_triggered': app_state.reset_triggered,
            'points_captured': app_state.points_captured.copy()
        }
    except Exception as e:
        print(f"[Flask] Error in get_command_state: {e}")
        return {
            'selected_object': 'tibia',
            'ct_side': None,
            'capture_point': False,
            'reset_triggered': False,
            'points_captured': {'femur': 0, 'tibia': 0}
        }

def reset_capture_flag():
    """Reset capture flag after processing"""
    app_state.capture_triggered = False
    app_state.is_processing = False
    
def reset_reset_flag():
    """Reset the reset flag after processing"""
    app_state.reset_triggered = False

def update_point_count(obj, count):
    """Update point count for object"""
    try:
        app_state.points_captured[obj] = count
    except Exception as e:
        print(f"[Flask] Error in update_point_count: {e}")

def update_current_point(message):
    """Update current point message for UI display"""
    try:
        app_state.current_point_message = message
    except Exception as e:
        print(f"[Flask] Error in update_current_point: {e}")

def update_marker_status(calibrated, markers_data):
    """
    Update marker validation status.
    
    Args:
        calibrated (bool): Whether calibration is complete
        markers_data (dict): Dictionary with marker IDs as keys and status dict as values
    """
    try:
        app_state.marker_status['calibrated'] = calibrated
        
        for marker_id, status in markers_data.items():
            if marker_id in app_state.marker_status['markers']:
                app_state.marker_status['markers'][marker_id].update(status)
        
        # Debug output (reduced frequency)
        if calibrated and app_state.frame_count % 300 == 0:  # Every 300 frames
            detected_count = sum(1 for m in markers_data.values() if m.get('detected', False))
            valid_count = sum(1 for m in markers_data.values() if m.get('valid', False))
            print(f"[Marker Status] Calibrated: {calibrated} | Detected: {detected_count}/4 | Valid: {valid_count}/4")
    
    except Exception as e:
        print(f"[Flask] Error in update_marker_status: {e}")

def run_flask_server():
    """Run Flask server in thread"""
    print("[Flask] Starting server on http://0.0.0.0:5000")
    print("[Flask] MJPEG stream: http://0.0.0.0:5000/video_feed")
    print("[Flask] Marker status tracking enabled")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)

if __name__ == '__main__':
    run_flask_server()
