# Correctiond on masks

import cv2
import numpy as np
from scipy.ndimage import label, find_objects

def keep_largest_object(mask):
    """
    Keep only the largest object in a binary segmentation mask.

    Parameters:
    mask (np.ndarray): A binary segmentation mask (2D array) where the object is marked with 1s and the background with 0s.

    Returns:
    np.ndarray: A binary segmentation mask with only the largest object.
    """
    # Label connected components
    labeled_mask, num_features = label(mask)

    if num_features == 0:
        return mask  # No objects found

    # Find the size of each connected component
    component_sizes = np.bincount(labeled_mask.ravel())

    # Ignore the background component (label 0)
    component_sizes[0] = 0

    # Find the largest component
    largest_component = component_sizes.argmax()

    # Create a mask with only the largest component
    largest_mask = (labeled_mask == largest_component)

    return largest_mask.astype(np.uint8)  # Ensure the mask is binary (0s and 1s)

def main(input_path, output_path):
    """
    Load a binary segmentation mask, keep only the largest object, and save the result.

    Parameters:
    input_path (str): Path to the input image file.
    output_path (str): Path to save the output image file.
    """
    # Load the binary segmentation mask
    mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    _,mask = cv2.threshold(mask, 127,255, cv2.THRESH_BINARY)

    # Ensure the mask is binary
    #mask = (mask > 0).astype(np.uint8)
    

    # Keep only the largest object
    largest_object_mask = keep_largest_object(mask)

    # Save the result
    cv2.imwrite(output_path, largest_object_mask*255)  # Multiply by 255 to make the mask visible

if __name__ == "__main__":
    input_path = r"C:\Users\chapi\Documents\STAGE\CODE\bgslibrary\ACtoolbox-main\Dataset\Small_ARIS_Mauzac\TEST\All_Originals\NewMasks2\m_Eel_Arcing_2014-11-16_020000_t12_Obj_frame2364.jpg"  # Replace with your input image path
    output_path = r"C:\Users\chapi\Documents\STAGE\CODE\bgslibrary\ACtoolbox-main\Dataset\Small_ARIS_Mauzac\TEST\All_Originals\NewMasks2\Corectedm_Eel_Arcing_2014-11-16_020000_t12_Obj_frame2364.jpg"  # Replace with your output image path
    main(input_path, output_path)
