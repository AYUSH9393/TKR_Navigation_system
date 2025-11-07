# app.py - Flask server to run on Raspberry Pi
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import threading
import time
import cv2
import numpy as np
from collections import deque
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Global variables shared between Flask and main application
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

app_state = AppState()

# ==================== ROUTES ====================

@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    """Return current system status"""
    return jsonify({
        'selected_object': app_state.selected_object,
        'ct_side': app_state.ct_side,
        'is_processing': app_state.is_processing,
        'points_captured': app_state.points_captured,
        'data_ready': app_state.data_ready
    })

@app.route('/api/frame', methods=['GET'])
def get_frame():
    """Return current camera frame as JPEG base64"""
    with app_state.frame_lock:
        if app_state.current_frame is not None:
            _, buffer = cv2.imencode('.jpg', app_state.current_frame)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            return jsonify({'frame': frame_b64})
    return jsonify({'error': 'No frame available'}), 400

@app.route('/api/results', methods=['GET'])
def get_results():
    """Return analysis results"""
    return jsonify(app_state.results)

@app.route('/api/command', methods=['POST'])
def send_command():
    """Handle commands from web interface"""
    data = request.json
    command = data.get('command')
    
    if command == 'select_femur':
        app_state.selected_object = "femur"
        app_state.points_captured['femur'] = 0
        app_state.messages.append("Femur selected")
    
    elif command == 'select_tibia':
        app_state.selected_object = "tibia"
        app_state.points_captured['tibia'] = 0
        app_state.messages.append("Tibia selected")
    
    elif command == 'set_ct_side':
        app_state.ct_side = data.get('side')  # 'L' or 'R'
        app_state.messages.append(f"CT Side: {app_state.ct_side}")
    
    elif command == 'capture_point':
        app_state.is_processing = True
        if app_state.selected_object == "femur":
            app_state.points_captured['femur'] += 1
        else:
            app_state.points_captured['tibia'] += 1
        app_state.messages.append(f"Point captured for {app_state.selected_object}")
        return jsonify({'status': 'Point capture initiated'})
    
    elif command == 'reset':
        obj = app_state.selected_object
        app_state.points_captured[obj] = 0
        app_state.messages.append(f"Reset {obj}")
    
    elif command == 'analyze':
        # Trigger analysis - this will be called when all points are captured
        app_state.data_ready = True
        app_state.messages.append("Analysis started")
    
    return jsonify({'status': 'Command received'})

@app.route('/api/messages', methods=['GET'])
def get_messages():
    """Get system messages"""
    messages = app_state.messages[-10:]  # Last 10 messages
    app_state.messages = []
    return jsonify({'messages': messages})

# ==================== DATA UPDATE FUNCTIONS ====================

def update_frame(frame):
    """Update current frame (call from main app)"""
    with app_state.frame_lock:
        app_state.current_frame = frame.copy()

def update_results(var_angle, var_flex, tmc_distance, tlc_distance, 
                   var_angle_name, var_flex_name):
    """Update analysis results (call from tibia_verifycuts)"""
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

def get_command_state():
    """Get state from web (call from main app)"""
    return {
        'selected_object': app_state.selected_object,
        'ct_side': app_state.ct_side,
        'capture_point': app_state.is_processing,
        'points_captured': app_state.points_captured
    }

def reset_capture_flag():
    """Reset capture flag after point is captured"""
    app_state.is_processing = False

# ==================== MAIN APP INTEGRATION ====================

if __name__ == '__main__':
    # Run Flask on all interfaces at port 5000
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)