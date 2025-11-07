import configparser

config = configparser.ConfigParser()
'''
#change according to your path
config["PATHS"] = {
    'ArucoImagePath' : '/Users/jaykuvadiya/DAIICT/BMP/Stereo_final/Aruco_Images/',
    'CalibrationImagePath' : 'E:/BMP/StereoCam/github/Calibration_Images/',
}

# change according to your image format
config["IMAGE_FORMAT"] = {
    'ArucoImageFormat' : '.jpeg',
    'CalibrationImageFormat' : '.jpeg',
}
'''
config["PATHS"] = {
    'ArucoImagePath' : 'C:/Users/Dell/Downloads/Final Stereo/Aruco_Images/Ayush/',
    'CalibrationImagePath' : 'C:/Users/Dell/Downloads/Final Stereo/Calibration_Images/Real_sense_10x7/1/',
}

# change according to your image format
config["IMAGE_FORMAT"] = {
    'ArucoImageFormat' : '.jpg',
    'CalibrationImageFormat' : '.jpg',
}

with open('config.ini', 'w') as configfile:
    config.write(configfile)
