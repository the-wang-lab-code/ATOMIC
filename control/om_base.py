import pythoncom
import win32com.client
import os
import numpy as np
import cv2
from PIL import Image
from aicspylibczi import CziFile
import base64
from pathlib import Path
from datetime import datetime
from kmeans_processor import process_image

class OM:
    def __init__(self):
        # Initialize the connection to the OM (ZEISS ZEN)
        pythoncom.CoInitialize()
        self.Zen = win32com.client.GetActiveObject("Zeiss.Micro.Scripting.ZenWrapperLM")
        self.image_save_dir = 'save_direction'
        os.makedirs(self.image_save_dir, exist_ok=True)

    def start_live_imaging(self):
        self.Zen.Acquisition.StartLive()
        print("Live imaging started.")

    def get_current_position(self):
        current_x = self.Zen.Devices.Stage.ActualPositionX
        current_y = self.Zen.Devices.Stage.ActualPositionY
        print(f"Current Position -> X: {current_x}, Y: {current_y}")
        return current_x, current_y

    def auto_focus_and_exposure(self):
        try:
            self.Zen.Acquisition.AutoExposure_2()
            self.Zen.Acquisition.FindAutofocus()
            print("Auto exposure and autofocus completed.")
        except Exception as e:
            print(f"Failed to perform auto exposure or autofocus: {e}")

    def auto_adjustment(self, X, Y):
        try:
            self.Zen.Devices.Stage.TargetPositionX = X
            self.Zen.Devices.Stage.TargetPositionY = Y
            self.Zen.Devices.Stage.Apply()
            print("Autoadjustment completed.")
        except Exception as e:
            print(f"Autoadjustment failed: {e}")

    def auto_adjustment_relative(self, x, y):
        try:
            self.Zen.Devices.Stage.TargetPositionX += x
            self.Zen.Devices.Stage.TargetPositionY += y
            self.Zen.Devices.Stage.Apply()
            print("Relative autoadjustment completed.")
        except Exception as e:
            print(f"Relative autoadjustment failed: {e}")

    def capture_and_save_image(self):
        """Capture an image and save it in .czi format."""
        try:
            image = self.Zen.Acquisition.AcquireImage()
            self.Zen.Application.Documents.Add(image)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = self.image_save_dir + f"//captured_image_{timestamp}.jpg"
            self.Zen.Application.Save(image, file_path, False)
            print(f"Image saved to: {file_path}")
            return file_path
        except Exception as e:
            print(f"Failed to capture image: {e}")
            return None
    
    def search_area(self, image_save_dir='save_direction', min_size=200, magnification=10, area_size_x=3000, area_size_y=3000, x_step=1216, y_step=1028):
        origin_x, origin_y = self.get_current_position()

        relative_x = 0
        relative_y = 0

        moving_left = True 

        while relative_y < area_size_y:
            while (moving_left and relative_x < area_size_x) or (not moving_left and relative_x > 0):
                self.auto_focus_and_exposure()
                img_dir = self.capture_and_save_image()
                process_image(img_dir, min_size, magnification)

                if moving_left:
                    relative_x += x_step  
                    self.Zen.Devices.Stage.TargetPositionX = origin_x + relative_x
                else:
                    relative_x -= x_step  
                    self.Zen.Devices.Stage.TargetPositionX = origin_x + relative_x
                self.Zen.Devices.Stage.Apply()

            relative_y += y_step
            self.Zen.Devices.Stage.TargetPositionY = origin_y + relative_y
            self.Zen.Devices.Stage.Apply()

            moving_left = not moving_left

            if relative_y >= area_size_y:
                print("Reached the top edge. Search completed.")
                break


    def stop_live_imaging(self):
        self.Zen.Acquisition.StopLive()
        print("Live imaging stopped.")

    
    @staticmethod
    def adjust_gamma(image, gamma=1.0):
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)


class OM_Handler:
    """Singleton class to handle OM control."""
    _om_instance = None

    @classmethod
    def get_instance(cls):
        if cls._om_instance is None:
            cls._om_instance = OM()
        return cls._om_instance

if __name__ == "__main__":
    # Example usage
    om_handler = OM_Handler.get_instance()
    om_handler.start_live_imaging()
    om_handler.set_exposure(50)
    om_handler.find_autofocus()

    image_path = om_handler.capture_and_save_image()
    om_handler.stop_live_imaging()

    if image_path:
        image_path = om_handler.convert_czi_to_tiff_and_process(image_path)
        print(image_path)

        with open(image_path, "rb") as image_file:
            print(base64.b64encode(image_file.read()).decode('utf-8'))
