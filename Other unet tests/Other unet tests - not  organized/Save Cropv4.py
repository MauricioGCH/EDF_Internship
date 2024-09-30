import os
import glob
import cv2
import numpy as np
import re

def extract_frame_number(file_path):
    match = re.search(r'_frame(\d+)', file_path)
    if match:
        return int(match.group(1))
    return -1  # Default value if no frame number is found

# Function to find the bounding box of non-zero values in the mask
def get_bounding_box(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop(mask, rmin, rmax, cmin, cmax, window_height, window_width):
    obj_height = rmax - rmin + 1
    obj_width = cmax - cmin + 1

    # Calculate the center of the object
    center_r = (rmin + rmax) // 2
    center_c = (cmin + cmax) // 2

    half_window_height = window_height // 2
    half_window_width = window_width // 2

    # Adjust start and end coordinates based on the center and window size
    start_r = max(center_r - half_window_height, 0)
    start_c = max(center_c - half_window_width, 0)
    end_r = min(center_r + half_window_height, mask.shape[0])
    end_c = min(center_c + half_window_width, mask.shape[1])

    # Adjust the edges if the window goes out of image bounds
    if start_r == 0:
        end_r = min(window_height, mask.shape[0])
    if start_c == 0:
        end_c = min(window_width, mask.shape[1])
    if end_r == mask.shape[0]:
        start_r = max(0, mask.shape[0] - window_height)
    if end_c == mask.shape[1]:
        start_c = max(0, mask.shape[1] - window_width)

    cropped = mask[int(start_r):int(end_r), int(start_c):int(end_c)]

    return cropped

# Function to process videos in a given dataset type
def process_videos(dataset_type, window_height, window_width):
    dataset_path = os.path.join(r'Small_ARIS_Mauzac_UnetReady_Final', dataset_type)
    videos = glob.glob(os.path.join(dataset_path, '2014*'))

    for video_path in videos:
        masks_paths = glob.glob(os.path.join(video_path, "Foreground", "t*"))
        originals_paths = glob.glob(os.path.join(video_path, "Original", "t*"))

        if len(masks_paths) != len(originals_paths):
            print(f"Error : Video in {dataset_type} doesn't have the same tracks for gt and input")
            break

        for i in range(len(masks_paths)):
            masks_in_track = glob.glob(os.path.join(masks_paths[i], "m_*"))
            sorted_masks_in_track = sorted(masks_in_track, key=extract_frame_number)

            originals_in_track = glob.glob(os.path.join(originals_paths[i], "2014*"))
            sorted_originals_in_track = sorted(originals_in_track, key=extract_frame_number)

            if len(sorted_masks_in_track) != len(sorted_originals_in_track):
                print(f"Error : Video in {dataset_type} doesn't have the same tracks for gt and input")
                break

            for j in range(len(sorted_masks_in_track)):
                mask_filename = os.path.basename(sorted_masks_in_track[j])
                mask_dir = os.path.dirname(sorted_masks_in_track[j])

                # Load the mask and original image
                mask = cv2.imread(sorted_masks_in_track[j], cv2.IMREAD_GRAYSCALE)
                original_image = cv2.imread(sorted_originals_in_track[j])

                # Get the bounding box containing all non-zero values
                rmin, rmax, cmin, cmax = get_bounding_box(mask)

                # Crop the mask and original image
                resized_mask = crop(mask, rmin, rmax, cmin, cmax, window_height, window_width)
                resized_image = crop(original_image, rmin, rmax, cmin, cmax, window_height, window_width)

                # Output paths for cropped images
                save_base_path = os.path.join(r"..\Pytorch-UNet-masterv2", f'N{window_height}x{window_width}\data_{dataset_type.lower()}')
                masks_save_path = os.path.join(save_base_path, "masks")
                images_save_path = os.path.join(save_base_path, "imgs")
                
                os.makedirs(masks_save_path, exist_ok=True)
                os.makedirs(images_save_path, exist_ok=True)

                cropped_mask_path = os.path.join(masks_save_path, f'crop_{mask_filename}')
                cropped_original_path = os.path.join(images_save_path, f'crop_{os.path.basename(sorted_originals_in_track[j])}')

                # Save the cropped images
                cv2.imwrite(cropped_mask_path, resized_mask)
                cv2.imwrite(cropped_original_path, resized_image)

                print(f'Saved cropped mask to: {cropped_mask_path}')
                print(f'Saved cropped original image to: {cropped_original_path}')

# List of dataset types
dataset_types = ['Train', 'Val', 'Test']

# Define the crop dimensions
window_height = 1276
window_width = 664

# Process each dataset type
for dataset_type in dataset_types:
    process_videos(dataset_type, window_height, window_width)

