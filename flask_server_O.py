# flask_server.py - Save this as a separate file
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import threading
import time
import cv2
import numpy as np
import base64

app = Flask(__name__)
CORS(app)

class AppState:
    def __init__(self):
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.selected_object = "femur"
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

app_state = AppState()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        'selected_object': app_state.selected_object,
        'ct_side': app_state.ct_side,
        'is_processing': app_state.is_processing,
        'points_captured': app_state.points_captured,
        'data_ready': app_state.data_ready
    })

@app.route('/api/frame', methods=['GET'])
def get_frame():
    with app_state.frame_lock:
        if app_state.current_frame is not None and app_state.current_frame.size > 0:
            try:
                # Resize for faster transmission if needed
                height, width = app_state.current_frame.shape[:2]
                if width > 640:
                    scale = 640 / width
                    new_width = 640
                    new_height = int(height * scale)
                    frame_resized = cv2.resize(app_state.current_frame, (new_width, new_height))
                else:
                    frame_resized = app_state.current_frame
                
                # Encode as JPEG
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                success, buffer = cv2.imencode('.jpg', frame_resized, encode_param)
                
                if success:
                    frame_b64 = base64.b64encode(buffer).decode('utf-8')
                    return jsonify({'frame': frame_b64})
                else:
                    return jsonify({'error': 'Encoding failed'}), 500
            except Exception as e:
                print(f"[Frame Encode Error]: {e}")
                return jsonify({'error': str(e)}), 500
    
    print("[Warning] No frame available in app_state")
    return jsonify({'error': 'No frame available'}), 400

@app.route('/api/results', methods=['GET'])
def get_results():
    return jsonify(app_state.results)

@app.route('/api/command', methods=['POST'])
def send_command():
    data = request.json
    command = data.get('command')
    
    if command == 'select_femur':
        app_state.selected_object = "femur"
        app_state.messages.append("Femur selected")
    
    elif command == 'select_tibia':
        app_state.selected_object = "tibia"
        app_state.messages.append("Tibia selected")
    
    elif command == 'set_ct_side':
        app_state.ct_side = data.get('side')
        app_state.messages.append(f"CT Side: {app_state.ct_side}")
    
    elif command == 'capture_point':
        app_state.capture_triggered = True
        app_state.is_processing = True
        app_state.messages.append("Capture initiated")
    
    elif command == 'reset':
        obj = app_state.selected_object
        app_state.points_captured[obj] = 0
        app_state.messages.append(f"Reset {obj}")
    
    elif command == 'analyze':
        app_state.data_ready = True
        app_state.messages.append("Analysis started")
    
    return jsonify({'status': 'Command received'})

@app.route('/api/messages', methods=['GET'])
def get_messages():
    messages = app_state.messages[-10:]
    app_state.messages = []
    return jsonify({'messages': messages})

# Helper functions to be imported by main_keyboard.py
def update_frame(frame):
    """Update current frame"""
    if frame is None or frame.size == 0:
        print("[Warning] Received empty frame")
        return
    
    with app_state.frame_lock:
        app_state.current_frame = frame.copy()
        # Debug output every 100 frames
        if not hasattr(update_frame, 'count'):
            update_frame.count = 0
        update_frame.count += 1
        if update_frame.count % 100 == 0:
            print(f"[Flask] Frame {update_frame.count} updated: {frame.shape}")

def update_results(var_angle, var_flex, tmc_distance, tlc_distance, 
                   var_angle_name, var_flex_name):
    """Update analysis results"""
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
    print(f"[Flask] Results updated: {var_angle_name} {abs(var_angle):.1f}°")

def get_command_state():
    """Get current state from web"""
    return {
        'selected_object': app_state.selected_object,
        'ct_side': app_state.ct_side,
        'capture_point': app_state.capture_triggered,
        'points_captured': app_state.points_captured.copy()
    }

def reset_capture_flag():
    """Reset capture flag after processing"""
    app_state.capture_triggered = False
    app_state.is_processing = False

def update_point_count(obj, count):
    """Update point count for object"""
    app_state.points_captured[obj] = count

def run_flask_server():
    """Run Flask server in thread"""
    print("[Flask] Starting server on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)

if __name__ == '__main__':
    run_flask_server()