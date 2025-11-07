import numpy as np
import cv2
import glob
import argparse


class StereoCalibration(object):
    def __init__(self, filepath, img_format='.jpg'):
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS +
                             cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((10*7, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:10, 0:7].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane. 

        self.cal_path = filepath
        self.read_images(self.cal_path, img_format)
        print(self.cal_path)
        print(img_format)
        # print("IN INIT")

    def read_images(self, cal_path, img_format):
        #images_right = glob.glob(cal_path + 'RIGHT/*.JPG')
        #images_left = glob.glob(cal_path + 'LEFT/*.JPG')
        print(cal_path + 'RIGHT/*' + img_format)
        images_right = glob.glob(cal_path + 'RIGHT/*' + img_format)
        images_left = glob.glob(cal_path + 'LEFT/*'   + img_format)
        
        images_left.sort()
        images_right.sort()
        
        # print(images_left)
        # print(images_right)
        
        

        for i, fname in enumerate(images_right):
            # print(images_left[i])
            # print(images_left[i])
            # print('********************************************')
            img_l = cv2.imread(images_left[i])
            img_r = cv2.imread(images_right[i])

            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, (10, 7), None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, (10, 7), None)
            
            print(ret_l, ret_r)
            # print(f"Left Corners: {corners_l.shape}")
            # print(f"Right Corners: {corners_r.shape}")

            # If found, add object points, image points (after refining them)
            if ret_l is True and ret_r is True:
                self.objpoints.append(self.objp)

            if ret_l is True and ret_r is True:
                rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_l.append(corners_l)
                print("corners_l")

                # Draw and display the corners
                ret_l = cv2.drawChessboardCorners(img_l, (10, 7),
                                                  corners_l, ret_l)
                #cv2.imshow(images_left[i], img_l)
                cv2.waitKey(500)
            
            #if ret_r is True and ret_l is True:
                rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_r.append(corners_r)
                print("corners_r")
                # Draw and display the corners
                ret_r = cv2.drawChessboardCorners(img_r, (10, 7),
                                                  corners_r, ret_r)
                #cv2.imshow(images_right[i], img_r)
                cv2.waitKey(500)
            img_shape = gray_l.shape[::-1]
            print(i)
            print("Image Shape:", img_shape)

        #print(self.objpoints)
        cv2.destroyAllWindows()
        print("Image reading Done")
        print(f"Number of Object Points: {len(self.objpoints)}")
        print(f"Number of Image Points (Left): {len(self.imgpoints_l)}")
        print(f"Number of Image Points (Right): {len(self.imgpoints_r)}")
        
        rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, img_shape, None, None)
        
        print("left calibration done")
        
        rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, img_shape, None, None)
        print("right calibration done")
        self.camera_model = self.stereo_calibrate(img_shape)

    def stereo_calibrate(self, dims):
        print("in stereo calibration")
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5

        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l,
            self.imgpoints_r, self.M1, self.d1, self.M2,
            self.d2, dims,
            criteria=stereocalib_criteria, flags=flags)
        print("Ret : " , ret)
        print('Intrinsic_mtx_1', M1)
        print('dist_1', d1)
        print('Intrinsic_mtx_2', M2)
        print('dist_2', d2)
        print('R', R)
        print('T', T)
        print('E', E)
        print('F', F)

        # for i in range(len(self.r1)):
        #     print("--- pose[", i+1, "] ---")
        #     self.ext1, _ = cv2.Rodrigues(self.r1[i])
        #     self.ext2, _ = cv2.Rodrigues(self.r2[i])
        #     print('Ext1', self.ext1)
        #     print('Ext2', self.ext2)

        print('')

        camera_model = dict([('M1', M1), ('M2', M2), ('dist1', d1),
                            ('dist2', d2), ('rvecs1', self.r1),
                            ('rvecs2', self.r2), ('R', R), ('T', T),
                            ('E', E), ('F', F)])
        print("Left Camera Matrix (M1):", self.M1)
        print("Left Distortion Coefficients (d1):", self.d1)
        print("Right Camera Matrix (M2):", self.M2)
        print("Right Distortion Coefficients (d2):", self.d2)


        cv2.destroyAllWindows()
        #print(camera_model)
        return camera_model
    # def stereo_calibrate(self, dims):
    #     print("in stereo calibration")

    #     # Using hardcoded values for calibration
    #     ret = 94.958
    #     M1 = [[0.499, 0.799, 0.498], [0.490, -0.057, 0.067], [0.0, -0.001, -0.021]]
    #     d1 = [[-0.0336802, -0.00282186, -0.00036508, -0.00209453, 0.01119475]]
    #     M2 = [[0.502, 0.802, 0.502], [0.492, -0.059, 0.067], [0.0, 0.0, -0.022]]
    #     d2 = [[-0.04201758, 0.01759639, -0.00091204, -0.00254141, -0.00097187]]
    #     R = [[1.0, -0.003, 0.009], [0.003, 1.0, 0.001], [-0.009, -0.001, 1.0]]
    #     T = [[-0.25342741], [-0.01643898], [0.19827329]]
    #     E = [[-0.00114481, -0.19830926, -0.01595815], 
    #         [0.22663176, 0.00131883, 0.22841641], 
    #         [0.01732695, -0.25336405, -0.00145909]]
    #     F = [[-1.10907055e-07, -1.90127638e-05, 5.85635731e-03], 
    #         [2.17149626e-05, 1.25055793e-07, -3.99182438e-04], 
    #         [-6.62926047e-03, -2.88995640e-03, 1.00000000e+00]]

    #     # Display the calibration data
    #     print(f"Ret :  {ret:.6f}")
    #     print(f"Intrinsic_mtx_1 {format_matrix(M1)}")
    #     print(f"dist_1 {format_matrix(d1)}")
    #     print(f"Intrinsic_mtx_2 {format_matrix(M2)}")
    #     print(f"dist_2 {format_matrix(d2)}")
    #     print(f"R {format_matrix(R)}")
    #     print(f"T {format_matrix(T)}")
    #     print(f"E {format_matrix(E)}")
    #     print(f"F {format_matrix(F)}")

    #     # Optional: Store values into a dictionary (if needed)
    #     camera_model = dict([('M1', M1), ('M2', M2), ('dist1', d1),
    #                         ('dist2', d2), ('R', R), ('T', T),
    #                         ('E', E), ('F', F)])
        
    #     print("\nLeft Camera Matrix (M1):", format_matrix(M1))
    #     print("Left Distortion Coefficients (d1):", format_matrix(d1))
    #     print("Right Camera Matrix (M2):", format_matrix(M2))
    #     print("Right Distortion Coefficients (d2):", format_matrix(d2))

        


    #     cv2.destroyAllWindows()

    #     return camera_model

    
#example
# filepath = 'C:/Users/ravik/OneDrive/Desktop/Ayush_A3D/New_Stereo_Cam/calibration_images/'
# img_format = '.jpg'
# cal = StereoCalibration(filepath, img_format)
