import cv2
import numpy as np

def process_binary_mask(input_file, output_file):
    # Load the binary mask image
    binary_mask = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)

    if binary_mask is None:
        print(f"Error: Unable to load image '{input_file}'.")
        return

    # Perform morphological opening to remove noise and smooth edges
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours of connected components
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask to keep only the largest region
    mask = np.zeros_like(binary_mask)
    max_area = 0
    best_contour = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            best_contour = contour

    if best_contour is not None:
        # Draw the largest contour on the mask
        cv2.drawContours(mask, [best_contour], -1, 255, thickness=cv2.FILLED)

        # Apply the mask to the original binary mask image
        result_mask = cv2.bitwise_and(binary_mask, mask)

        # Save the result
        cv2.imwrite(output_file, result_mask)

        # Display the result (optional)
        cv2.imshow('Result', result_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: No contours found.")

# Example usage:
input_file = r"C:\Users\chapi\Documents\STAGE\CODE\bgslibrary\ACtoolbox-main\Dataset\Small_ARIS_Mauzac\TEST\All_Originals\NewMasks2\m_Eel_Arcing_2014-11-16_020000_t12_Obj_frame2361.jpg"
output_file = r"C:\Users\chapi\Documents\STAGE\CODE\bgslibrary\ACtoolbox-main\Dataset\Small_ARIS_Mauzac\TEST\All_Originals\NewMasks2\Correctedm_Eel_Arcing_2014-11-16_020000_t12_Obj_frame2361.jpg"
process_binary_mask(input_file, output_file)
