import cv2
from time import sleep

from om_base import OM, OM_Handler
from sam4segmentation import process_image_with_sam
from kmeans_processor import process_image

def move_to(
        X: float, 
        Y: float
):
    """
    OM Action. Move to a position specified by absolute coordinates, the origin is the right bottom corner of the stage, the x-axis is the horizontal direction with positive values to the left, and the y-axis is the vertical direction with positive values to the top. The range of X is 0 to 85000.00, the range of Y is 0 to 65000.00.

    
    Args:
        X: Stage position in absolute coordinates (in micrometers)
        Y: Stage position in absolute coordinates (in micrometers)
        
    Returns:
        str: Success message indicating the target coordinates.
    """
    om = OM_Handler.get_instance()
    try:
        om.auto_adjustment(X, Y)
        sleep(2)
        return f"Move to ({X}, {Y}) meters successfully!"
    except Exception as e:
        return f"Failed to move to ({X}, {Y}): {e}"
    
def move_by(
        x: float, 
        y: float
):
    """
    OM Action. Move to a position specified by relative coordinates
    
    Args:
        x: Move to a position specified relative to the current position (in micrometers), a positive x value will move to the left direction
        y: Move to a position specified relative to the current position (in micrometers), a positive y value will move to the up direction
        
    Returns:
        str: Success message indicating the target coordinates.
    """
    om = OM_Handler.get_instance()
    try:
        om.auto_adjustment_relative(x, y)
        sleep(2)
        return f"Move to ({x}, {y}) meters successfully!"
    except Exception as e:
        return f"Failed to move to ({x}, {y}): {e}"

def get_current_position():
    """
    OM Action. Get current position of the lens.
    
    Args:
        none
        
    Returns:
        str: Success message indicating current position detected.
    """
    om = OM_Handler.get_instance()
    try:
        om.get_current_position()
        sleep(2)
        print("Current position detected.")
    except Exception as e:
        print(f"Failed to detect current position: {e}")

def auto_focus_and_exposure():
    """
    OM Action. Automatically find the focus using the microscope's autofocus function, set the exposure time for the microscope camera.
    
    Args:
        exposure_time: Desired exposure time in milliseconds, often set under 80.
        
    Returns:
        str: Success message indicating autofocus was completed and with the exposure time set.
    """
    om = OM_Handler.get_instance()
    try:
        om.auto_focus_and_exposure()
        sleep(2)
        print("Auto exposure and autofocus completed.")
    except Exception as e:
        print(f"Failed to perform auto exposure or autofocus: {e}")

def area_analysis(
        area_size_x: float =3000, 
        area_size_y: float =3000, 
        x_step: float =1216, 
        y_step: float =1028
):
    """
    OM Action. Move over a designated area, capture, save and analyze image after each move, the analyze will find 2D materials larger than 200um.
    
    Args:
        area_size_x: x length of the designated area
        area_size_y: y length of the designated area
        x_step: x step size, default value is 1216
        y_step: y step size, default value is 1028

    Returns:
        str: Success message indicating the target coordinates.
    """
    om = OM_Handler.get_instance()
    try:
        om.search_area()
        sleep(2)
        return f"Area analysis completed."
    except Exception as e:
        return f"Area analysis failed."

def adjust_image_gamma(image_path: str, gamma: float):
    """
    Adjust the gamma of an image and save it.
    
    Args:
        image_path: Path to the image.
        gamma: Gamma correction value.
        
    Returns:
        str: Path to the gamma-adjusted image.
    """
    om = OM_Handler.get_instance()
    try:
        img = cv2.imread(image_path)
        if img is not None:
            processed_img = om.adjust_gamma(img, gamma)
            output_path = image_path.replace('.jpg', f'_gamma_{gamma}.jpg')
            cv2.imwrite(output_path, processed_img)
            return f"Gamma-adjusted image saved to: {output_path}"
        else:
            return "Error: Unable to read the image for gamma adjustment."
    except Exception as e:
        return f"Failed to adjust gamma: {e}"

def image_analysis(
        prompt: str
):
    """
    Analyze the image after operating the OM using SAM for segmentation. After processing, gives the proportion of non-substrate area.

    Args:
        prompt: The message to convey to the visual agent, including final objective, current state, and current task info.
                e.g. Final objective: xxx. Current state: Searching state. Current task: xxx.

    Returns:
        res: A detailed description of the image based on SAM segmentation
    """

    om = OM_Handler.get_instance()
    image_path = om.capture_and_save_image()

    sam_result = process_image_with_sam(image_path)

    res = f"Image analysis completed. Closest cluster to the substrate: {sam_result['closest_cluster']}, " \
          f"Cluster areas: {sam_result['cluster_areas']}, " \
          f"Non-substrate area proportion: {sam_result['non_substrate_area_proportion']}"

    return res

def image_analysis_kmeans(
        prompt: str,
        min_size: float,
        magnification: int
):
    """
    Analyze the image after operating the OM using the updated image processor for segmentation.

    Args:
        prompt: The message to convey to the visual agent, including final objective, current state, and current task info.
                e.g. Final objective: xxx. Current state: Searching state. Current task: xxx.
        min_size: The minimum size (in µm) for line dimensions to be considered valid.
        magnification: The magnification of the optical microscope (e.g., 10, 50, 100).

    Returns:
        res: A detailed description of the image based on the new image processor segmentation.
    """

    om = OM_Handler.get_instance()
    image_path = om.capture_and_save_image()

    # Process the image using the new image processor
    classifications, valid_regions = process_image(image_path, min_size, magnification)

    # Format the result to return
    res = f"Image analysis completed. Classifications: {classifications}, " \
          f"Valid regions for 2D material: {valid_regions}"

    return res



__functions__ = [
    move_to,
    move_by,
    get_current_position,
    auto_focus_and_exposure,
    area_analysis, 
    adjust_image_gamma,
    image_analysis,
]
