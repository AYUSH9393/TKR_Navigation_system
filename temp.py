from save_images import SaveImages
from CAMERA_CAPTURE_MODULE_MAIN import ComeraCapture_RealSense
import cv2
import time
# saving Calibration images
 
obj = ComeraCapture_RealSense(CAMERA_CONFIG_OBJ=None)

i = 0
while True:
    ret, L_img, R_img = obj.get_IR_FRAME_SET()
    if ret:
        cv2.imshow('L_img', L_img)
        cv2.imwrite(f"E:/BMP/StereoCam/github/Aruco_Test/2_CM/LEFT/L_img_{i}.png", L_img)
        cv2.imshow('R_img', R_img)
        cv2.imwrite(f"E:/BMP/StereoCam/github/Aruco_Test/2_CM/RIGHT/R_img_{i}.png", R_img)
        ## Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1
        time.sleep(0.5)