#Correction masks2 
import cv2
import numpy as np
import glob
import os
from skimage.measure import label,regionprops
import matplotlib.pyplot as plt
import scipy

def load_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def save_image(path, image):
    cv2.imwrite(path, image)

def keep_largest_connected_component(mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # If there's only one component (background) or no foreground, return the original mask
    if num_labels <= 1:
        return mask
    
    # Get the size of each component (except the background component 0)
    sizes = stats[1:, cv2.CC_STAT_AREA]
    
    # Find the largest component
    largest_label = 1 + np.argmax(sizes)
    
    # Create an output image where only the largest component is kept
    largest_component_mask = np.zeros_like(mask)
    largest_component_mask[labels == largest_label] = 255
    
    return largest_component_mask

def connect_regions(mask):
    Dilatatekernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
    Erodekernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    while True:
        # Perform dilation
        dilated_mask = cv2.dilate(mask, Dilatatekernel, iterations=1)
        
        # Perform erosion
        eroded_mask = cv2.erode(dilated_mask, Erodekernel, iterations=1)
        
        # Check the number of connected components
        num_labels, labels = cv2.connectedComponents(eroded_mask, connectivity=4)
        
        if num_labels <= 2:  # 1 for background and 1 for the object
            kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            final = cv2.dilate(eroded_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1)), iterations=1)
            kernel_close = np.ones((5, 5), np.uint8)
            final = scipy.ndimage.binary_closing(final, structure=kernel_close, iterations=2)*255
            return final
        mask = eroded_mask

def main():
    input_path = r"C:\Users\chapi\Documents\STAGE\CODE\bgslibrary\ACtoolbox-main\Dataset\Small_ARIS_Mauzac\TEST\All_Originals\NewMasks2\m_Eel_Arcing_2014-11-16_020000_t12_Obj_frame2354.jpg"  # Replace with your input image path
    output_path = r"C:\Users\chapi\Documents\STAGE\CODE\bgslibrary\ACtoolbox-main\Dataset\Small_ARIS_Mauzac\TEST\All_Originals\NewMasks2\Corectedm_Eel_Arcing_2014-11-16_020000_t12_Obj_frame2354.jpg"
    
    
    mask = load_image(input_path)
    #largest_component_mask = keep_largest_connected_component(mask)
    connected_mask = connect_regions(mask)
    save_image(output_path, connected_mask)
    
    # Display the result
    plt.imshow(connected_mask, cmap='gray')
    plt.title('Connected Mask')
    plt.show()

if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     input_path = r"C:\Users\chapi\Documents\STAGE\CODE\bgslibrary\ACtoolbox-main\Dataset\Small_ARIS_Mauzac\TEST\All_Originals\NewMasks2\m_Eel_Arcing_2014-11-16_020000_t12_Obj_frame2359.jpg"  # Replace with your input image path
#     output_path = r"C:\Users\chapi\Documents\STAGE\CODE\bgslibrary\ACtoolbox-main\Dataset\Small_ARIS_Mauzac\TEST\All_Originals\NewMasks2\Corectedm_Eel_Arcing_2014-11-16_020000_t12_Obj_frame2359.jpg"
    
#     main(input_path, output_path)
    
    # input_path = r"C:\Users\chapi\Documents\STAGE\CODE\bgslibrary\ACtoolbox-main\Dataset\Small_ARIS_Mauzac\TEST\All_Originals\NewMasks2\m_Eel_Arcing_2014-11-16_020000_t12_Obj_frame*.jpg"  # Replace with your input image path
    # output_path = r"C:\Users\chapi\Documents\STAGE\CODE\bgslibrary\ACtoolbox-main\Dataset\Small_ARIS_Mauzac\TEST\All_Originals\NewMasks2\Corectedm_Eel_Arcing_2014-11-16_020000_t12_Obj_frame2354.jpg"
    # directory = r"C:\Users\chapi\Documents\STAGE\CODE\bgslibrary\ACtoolbox-main\Dataset\Small_ARIS_Mauzac\TEST\All_Originals\NewMasks2"
    # imagepaths = glob.glob(input_path)
    
    # for image in imagepaths:
    #     filename = os.path.basename(image)
    #     main(image, os.path.join(directory, str("corrected_"+filename)))
