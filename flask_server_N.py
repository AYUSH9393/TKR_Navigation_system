# flask_server.py - Optimized with MJPEG streaming
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
        self.frame_queue = queue.Queue(maxsize=2)  # Small queue for latest frames
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
        self.frame_count = 0

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

# ==================== NEW MJPEG STREAMING ROUTE ====================
@app.route('/video_feed')
def video_feed():
    """
    MJPEG streaming route for smooth video playback.
    This is much more efficient than base64 encoding.
    """
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    """
    Generator function that yields JPEG frames for MJPEG streaming.
    Uses the frame queue for non-blocking frame retrieval.
    """
    while True:
        try:
            # Try to get frame from queue (non-blocking)
            if not app_state.frame_queue.empty():
                frame = app_state.frame_queue.get_nowait()
                
                if frame is not None and frame.size > 0:
                    # Encode as JPEG with good quality
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                    success, buffer = cv2.imencode('.jpg', frame, encode_param)
                    
                    if success:
                        # Yield frame in MJPEG format
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + 
                               buffer.tobytes() + b'\r\n')
            
            # Small sleep to prevent CPU overload
            time.sleep(0.03)  # ~30 FPS max
            
        except queue.Empty:
            time.sleep(0.01)
        except Exception as e:
            print(f"[MJPEG] Frame generation error: {e}")
            time.sleep(0.1)

# ==================== LEGACY BASE64 ENDPOINT (kept for compatibility) ====================
@app.route('/api/frame', methods=['GET'])
def get_frame():
    """
    Legacy base64 frame endpoint. 
    NOTE: Use /video_feed for better performance!
    """
    with app_state.frame_lock:
        if app_state.current_frame is not None and app_state.current_frame.size > 0:
            try:
                # Resize for faster transmission
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

# ==================== HELPER FUNCTIONS ====================
def update_frame(frame):
    """
    Update current frame - now uses both lock and queue for efficiency.
    Queue is used for MJPEG streaming, lock for legacy base64 endpoint.
    """
    if frame is None or frame.size == 0:
        return
    
    # Update locked frame (for legacy API)
    with app_state.frame_lock:
        app_state.current_frame = frame.copy()
    
    # Update queue (for MJPEG streaming) - non-blocking
    try:
        if app_state.frame_queue.full():
            app_state.frame_queue.get_nowait()  # Remove old frame
        app_state.frame_queue.put_nowait(frame.copy())
        
        app_state.frame_count += 1
        if app_state.frame_count % 100 == 0:
            print(f"[Flask] Frame {app_state.frame_count} updated: {frame.shape}")
    except queue.Full:
        pass  # Skip if queue full
    except Exception as e:
        print(f"[Flask] Queue error: {e}")

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
    print("[Flask] MJPEG stream available at: http://0.0.0.0:5000/video_feed")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)

if __name__ == '__main__':
    run_flask_server()