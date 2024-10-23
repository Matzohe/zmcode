from src.CV.ClassHomeWork.CameraCalibrate.camera_calibrate import CameraCalibrate
from src.utils.utils import INIconfig

def camera_test():
    config = INIconfig()
    camera = CameraCalibrate(config)
    camera.run()
