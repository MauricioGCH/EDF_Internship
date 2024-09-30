

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob


# Function to find the bounding box of non-zero values in the mask
def get_bounding_box(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

# Function to crop and resize the window around the object
def crop_and_resize(mask, rmin, rmax, cmin, cmax):
    obj_height = rmax - rmin + 1
    obj_width = cmax - cmin + 1

    if obj_height > 150 or obj_width > 150:
        # If the object is larger than 150x150, use the full bounding box and resize
        cropped = mask[rmin:rmax+1, cmin:cmax+1]
        resized = cv2.resize(cropped, (150, 150))
    else:
        # If the object fits within a 150x150 window, create a centered window
        center_r = (rmin + rmax) // 2
        center_c = (cmin + cmax) // 2
        start_r = max(center_r - 75, 0)
        start_c = max(center_c - 75, 0)
        end_r = min(center_r + 75, mask.shape[0])
        end_c = min(center_c + 75, mask.shape[1])
        
        # Adjust the edges if the window goes out of image bounds
        if start_r == 0:
            end_r = min(150, mask.shape[0])
        if start_c == 0:
            end_c = min(150, mask.shape[1])
        if end_r == mask.shape[0]:
            start_r = max(0, mask.shape[0] - 150)
        if end_c == mask.shape[1]:
            start_c = max(0, mask.shape[1] - 150)
        
        cropped = mask[start_r:end_r, start_c:end_c]
        resized = cv2.resize(cropped, (150, 150))

    return resized

# List of image paths
Train = glob.glob(os.path.join(r'C:\Users\d42684\Documents\STAGE\CODES\Small_ARIS_Mauzac\Train\2014*\Foreground\t*\*'))
Val = glob.glob(os.path.join(r'C:\Users\d42684\Documents\STAGE\CODES\Small_ARIS_Mauzac\Val\2014*\Foreground\t*\*'))
Test = glob.glob(os.path.join(r'C:\Users\d42684\Documents\STAGE\CODES\Small_ARIS_Mauzac\Test\2014*\Foreground\t*\*'))

AllMaskImages = Train + Val + Test


TrainO = glob.glob(os.path.join(r'C:\Users\d42684\Documents\STAGE\CODES\Small_ARIS_Mauzac\Train\2014*\Original\t*\*'))
ValO = glob.glob(os.path.join(r'C:\Users\d42684\Documents\STAGE\CODES\Small_ARIS_Mauzac\Val\2014*\Original\t*\*'))
TestO = glob.glob(os.path.join(r'C:\Users\d42684\Documents\STAGE\CODES\Small_ARIS_Mauzac\Test\2014*\Original\t*\*'))

AllOriginalImages = TrainO + ValO + TestO

delay = 0.4
# Iterate over the provided image paths
for mask_filepath in AllMaskImages:
    if mask_filepath.endswith(('.png', '.jpg')):
        mask_filename = os.path.basename(mask_filepath)
        
        # Extract date and frame number from the mask filename
        mask_parts = mask_filename.split('_')
        date_part = mask_parts[2]
        frame_part = mask_parts[-1].split('.')[0].replace('frame', '')

        # Match with the corresponding original image
        matched_original = None
        for original_filepath in AllOriginalImages:
            original_filename = os.path.basename(original_filepath)
            if date_part in original_filename and frame_part in original_filename:
                matched_original = original_filepath
                break

        if matched_original:
            # Load the mask and original image
            mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
            original_image = cv2.imread(matched_original)

            # Get the bounding box containing all non-zero values
            rmin, rmax, cmin, cmax = get_bounding_box(mask)

            resized_mask = crop_and_resize(mask, rmin, rmax, cmin, cmax)
            resized_image = crop_and_resize(original_image, rmin, rmax, cmin, cmax)

            # Crop and resize the window around the object in the original image
            resized_image = crop_and_resize(original_image, rmin, rmax, cmin, cmax)

            # Display the mask and the resized image side by side
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 2, 1)
            plt.imshow(resized_mask, cmap='gray')
            plt.title(f'Cropped Mask: {mask_filename}')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
            plt.title(f'Cropped and Resized: {os.path.basename(matched_original)}')
            plt.axis('off')
            #plt.draw()
            
            
            # Wait for the specified delay before showing the next image
        
            plt.pause(delay)
            plt.clf()  
            




       
