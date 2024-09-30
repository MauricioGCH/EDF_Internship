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

# Function to find the window of fixed size with the most object regions
def get_best_window(mask, rmin, rmax, cmin, cmax, window_size):
    best_window = None
    max_white_pixels = 0

    for start_r in range(rmin, rmax - window_size + 2):
        for start_c in range(cmin, cmax - window_size + 2):
            end_r = start_r + window_size
            end_c = start_c + window_size
            window = mask[start_r:end_r, start_c:end_c]
            white_pixels = np.sum(window > 0)
            if white_pixels > max_white_pixels:
                max_white_pixels = white_pixels
                best_window = (start_r, end_r, start_c, end_c)

    if best_window is None:
        # Fallback to the default centered window
        center_r = (rmin + rmax) // 2
        center_c = (cmin + cmax) // 2
        start_r = max(center_r - window_size // 2, 0)
        start_c = max(center_c - window_size // 2, 0)
        end_r = min(start_r + window_size, mask.shape[0])
        end_c = min(start_c + window_size, mask.shape[1])
        best_window = (int(start_r), int(end_r), int(start_c), int(end_c))

    return best_window

# Function to crop the window around the object
def crop(mask, window):
    start_r, end_r, start_c, end_c = window
    cropped = mask[start_r:end_r, start_c:end_c]
    return cropped

# List of image paths
Train = glob.glob(os.path.join(r'Small_ARIS_Mauzac_UnetReady_Final\Train\2014*'))
AllVideos = Train

# Iterate over the provided image paths
for VideoPath in AllVideos:

    TrackspathsMasks = glob.glob(os.path.join(VideoPath, "Foreground", "t*"))
    TrackspathsOriginals = glob.glob(os.path.join(VideoPath, "Original", "t*"))

    if len(TrackspathsMasks) != len(TrackspathsOriginals):
        print("Error: Video doesn't have the same tracks for gt and input")
        break

    for i in range(len(TrackspathsMasks)):

        MasksInTrack = glob.glob(os.path.join(TrackspathsMasks[i], "m_*"))
        sorted_MasksInTrack = sorted(MasksInTrack, key=extract_frame_number)

        OriginalsInTrack = glob.glob(os.path.join(TrackspathsOriginals[i], "2014*"))
        sorted_OriginalsInTrack = sorted(OriginalsInTrack, key=extract_frame_number)

        if len(sorted_MasksInTrack) != len(sorted_OriginalsInTrack):
            print("Error: Video doesn't have the same tracks for gt and input")
            break

        for j in range(len(sorted_MasksInTrack)):

            mask_filename = os.path.basename(sorted_MasksInTrack[j])

            # Load the mask and original image
            mask = cv2.imread(sorted_MasksInTrack[j], cv2.IMREAD_GRAYSCALE)
            original_image = cv2.imread(sorted_OriginalsInTrack[j])

            # Get the bounding box containing all non-zero values
            rmin, rmax, cmin, cmax = get_bounding_box(mask)

            window_size = 350
            if (rmax - rmin + 1 > window_size) or (cmax - cmin + 1 > window_size):
                # Get the best window of size window_size that covers the most object regions
                window = get_best_window(mask, rmin, rmax, cmin, cmax, window_size)
            else:
                # Center the window around the object if it fits within the window size
                center_r = (rmin + rmax) // 2
                center_c = (cmin + cmax) // 2
                start_r = max(center_r - window_size // 2, 0)
                start_c = max(center_c - window_size // 2, 0)
                end_r = min(start_r + window_size, mask.shape[0])
                end_c = min(start_c + window_size, mask.shape[1])
                window = (int(start_r), int(end_r), int(start_c), int(end_c))

            # Crop and resize the window around the object in the mask
            resized_mask = crop(mask, window)

            # Crop and resize the window around the object in the original image
            resized_image = crop(original_image, window)

            # Output paths for cropped images
            os.makedirs(r"..\Pytorch-UNet-masterv2\N350x350\data_train\masks", exist_ok=True)
            os.makedirs(r"..\Pytorch-UNet-masterv2\N350x350\data_train\imgs", exist_ok=True)
            cropped_mask_path = os.path.join(r"..\Pytorch-UNet-masterv2\N350x350\data_train\masks", f'crop_{mask_filename}')
            cropped_original_path = os.path.join(r"..\Pytorch-UNet-masterv2\N350x350\data_train\imgs", f'crop_{os.path.basename(sorted_OriginalsInTrack[j])}')

            # Save the resized images with "crop_" prefix
            cv2.imwrite(cropped_mask_path, resized_mask)
            cv2.imwrite(cropped_original_path, resized_image)

            print(f'Saved cropped mask to: {cropped_mask_path}')
            print(f'Saved cropped original image to: {cropped_original_path}')
