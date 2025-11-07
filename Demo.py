import cv2
import numpy as np
import time

class StereoCameraProcessor:
    def __init__(self, rtmp_url="rtmp://10.42.0.1:1935/cam"):
        self.rtmp_url = rtmp_url
        self.cap = None
        self.is_stereo = False
        self.left_width = 0
        self.right_width = 0
        
    def initialize_camera(self):
        """Initialize camera connection"""
        print(f"🎥 Connecting to camera: {self.rtmp_url}")
        
        self.cap = cv2.VideoCapture(self.rtmp_url)
        
        # Set properties for better performance
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        self.cap.set(cv2.CAP_PROP_FPS, 30)        # Set FPS
        
        if not self.cap.isOpened():
            raise Exception("❌ Cannot connect to camera")
            
        # Test frame to determine if it's stereo
        ret, frame = self.cap.read()
        if ret:
            height, width, channels = frame.shape
            print(f"📐 Frame dimensions: {width}x{height} (channels: {channels})")
            
            # Check if it's likely a stereo camera (width > height significantly)
            if width > height * 2:
                self.is_stereo = True
                self.left_width = width // 2
                self.right_width = width - self.left_width
                print(f"🔍 Detected stereo camera: Left({self.left_width}px) + Right({self.right_width}px)")
            else:
                print("📷 Single camera detected")
                
            return True
        else:
            raise Exception("❌ Cannot read frames from camera")
    
    def split_stereo_frame(self, frame):
        """Split stereo frame into left and right images"""
        if not self.is_stereo:
            return frame, None
            
        height, width = frame.shape[:2]
        left_frame = frame[:, :self.left_width]
        right_frame = frame[:, self.left_width:]
        
        return left_frame, right_frame
    
    def process_frames(self):
        """Main processing loop"""
        if not self.cap or not self.cap.isOpened():
            self.initialize_camera()
            
        print("🚀 Starting frame processing...")
        print("Press 'q' to quit, 's' to save current frame")
        
        frame_count = 0
        fps_start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                print("⚠️ No frame received, retrying...")
                time.sleep(0.1)
                continue
                
            frame_count += 1
            
            # Calculate FPS every 30 frames
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start_time)
                print(f"📊 FPS: {fps:.2f}")
                fps_start_time = time.time()
            
            # Process stereo or single frame
            if self.is_stereo:
                left_frame, right_frame = self.split_stereo_frame(frame)
                
                # Example processing: Convert to grayscale
                left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
                
                # Example: Simple depth estimation (disparity)
                stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
                disparity = stereo.compute(left_gray, right_gray)
                
                # Normalize disparity for display
                disparity_display = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                
                # Display windows
                cv2.imshow('Left Camera', left_frame)
                cv2.imshow('Right Camera', right_frame)
                cv2.imshow('Left Gray', left_gray)
                cv2.imshow('Right Gray', right_gray)
                cv2.imshow('Disparity Map', disparity_display)
                
            else:
                # Single camera processing
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Example processing: Edge detection
                edges = cv2.Canny(gray, 50, 150)
                
                # Display windows
                cv2.imshow('Original Frame', frame)
                cv2.imshow('Grayscale', gray)
                cv2.imshow('Edges', edges)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("🛑 Stopping...")
                break
            elif key == ord('s'):
                timestamp = int(time.time())
                if self.is_stereo:
                    cv2.imwrite(f'left_frame_{timestamp}.jpg', left_frame)
                    cv2.imwrite(f'right_frame_{timestamp}.jpg', right_frame)
                    cv2.imwrite(f'disparity_{timestamp}.jpg', disparity_display)
                    print(f"💾 Saved stereo frames with timestamp {timestamp}")
                else:
                    cv2.imwrite(f'frame_{timestamp}.jpg', frame)
                    print(f"💾 Saved frame with timestamp {timestamp}")
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Cleanup completed")

# Example usage functions
def basic_camera_access():
    """Simple example of accessing the camera"""
    print("🔧 Basic Camera Access Example")
    print("=" * 40)
    
    cap = cv2.VideoCapture("rtmp://10.42.0.1:1935/cam")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if ret:
                # Your image processing here
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                cv2.imshow('Camera Feed', frame)
                cv2.imshow('Processed', gray)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("No frame received")
                time.sleep(0.1)
    
    cap.release()
    cv2.destroyAllWindows()

def stereo_vision_example():
    """Example of stereo vision processing"""
    print("🔍 Stereo Vision Processing Example")
    print("=" * 40)
    
    processor = StereoCameraProcessor()
    
    try:
        processor.initialize_camera()
        processor.process_frames()
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        processor.cleanup()

def custom_processing_template():
    """Template for custom image processing"""
    print("⚙️ Custom Processing Template")
    print("=" * 40)
    
    cap = cv2.VideoCapture("rtmp://10.42.0.1:1935/cam")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    while True:
        ret, frame = cap.read()
        if ret:
            # === YOUR CUSTOM PROCESSING HERE ===
            
            # Example 1: Split stereo frame
            height, width = frame.shape[:2]
            if width > height * 2:  # Likely stereo
                left = frame[:, :width//2]
                right = frame[:, width//2:]
            else:
                left = frame
                right = None
            
            # Example 2: Color detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Example 3: Object detection preparation
            # (You can add YOLO, detection models here)
            
            # Example 4: Feature detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
            
            if corners is not None:
                for corner in corners:
                    x, y = corner.ravel().astype(int)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            # Display results
            cv2.imshow('Original', frame)
            cv2.imshow('HSV', hsv)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Waiting for frame...")
            time.sleep(0.1)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("🎥 Stereo Camera OpenCV Processing")
    print("=" * 50)
    
    print("\nChoose processing mode:")
    print("1. Basic camera access")
    print("2. Full stereo vision processing")
    print("3. Custom processing template")
    
    choice = input("\nEnter choice (1-3) or press Enter for stereo processing: ").strip()
    
    if choice == "1":
        basic_camera_access()
    elif choice == "3":
        custom_processing_template()
    else:
        stereo_vision_example()