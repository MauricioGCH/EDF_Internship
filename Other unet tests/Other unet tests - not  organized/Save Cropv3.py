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

def crop(mask, rmin, rmax, cmin, cmax, window_size=500):
    obj_height = rmax - rmin + 1
    obj_width = cmax - cmin + 1

    # Calculate the center of the object
    center_r = (rmin + rmax) // 2
    center_c = (cmin + cmax) // 2

    half_window = window_size // 2

    # Adjust start and end coordinates based on the center and window size
    start_r = max(center_r - half_window, 0)
    start_c = max(center_c - half_window, 0)
    end_r = min(center_r + half_window, mask.shape[0])
    end_c = min(center_c + half_window, mask.shape[1])

    # Adjust the edges if the window goes out of image bounds
    if start_r == 0:
        end_r = min(window_size, mask.shape[0])
    if start_c == 0:
        end_c = min(window_size, mask.shape[1])
    if end_r == mask.shape[0]:
        start_r = max(0, mask.shape[0] - window_size)
    if end_c == mask.shape[1]:
        start_c = max(0, mask.shape[1] - window_size)

    cropped = mask[int(start_r):int(end_r), int(start_c):int(end_c)]

    return cropped

Train = glob.glob(os.path.join(r'Small_ARIS_Mauzac_UnetReady_Final\Train\2014*'))
AllVideos = Train

# Iterate over the provided image paths
for VideoPath in AllVideos:
    TrackspathsMasks = glob.glob(os.path.join(VideoPath,"Foreground","t*"))
    TrackspathsOriginals = glob.glob(os.path.join(VideoPath,"Original","t*"))

    if len(TrackspathsMasks) != len(TrackspathsOriginals):
        print("Error : Video doesn't have the same tracks for gt and input")
        break

    for i in range(len(TrackspathsMasks)):
        MasksInTrack = glob.glob(os.path.join(TrackspathsMasks[i],"m_*"))
        sorted_MasksInTrack = sorted(MasksInTrack, key=extract_frame_number)

        OriginalsInTrack = glob.glob(os.path.join(TrackspathsOriginals[i],"2014*"))
        sorted_OriginalsInTrack = sorted(OriginalsInTrack, key=extract_frame_number)

        if len(sorted_MasksInTrack) != len(sorted_OriginalsInTrack):
            print("Error : Video doesn't have the same tracks for gt and input")
            break

        for j in range(len(sorted_MasksInTrack)):
            mask_filename = os.path.basename(sorted_MasksInTrack[j])
            mask_dir = os.path.dirname(sorted_MasksInTrack[j])

            # Load the mask and original image
            mask = cv2.imread(sorted_MasksInTrack[j], cv2.IMREAD_GRAYSCALE)
            original_image = cv2.imread(sorted_OriginalsInTrack[j])

            # Get the bounding box containing all non-zero values
            rmin, rmax, cmin, cmax = get_bounding_box(mask)

            # Crop the mask and original image
            resized_mask = crop(mask, rmin, rmax, cmin, cmax, window_size=500)
            resized_image = crop(original_image, rmin, rmax, cmin, cmax, window_size=500)

            # Output paths for cropped images
            os.makedirs(r"..\Pytorch-UNet-masterv2\N500x500\data_train\masks", exist_ok=True)
            os.makedirs(r"..\Pytorch-UNet-masterv2\N500x500\data_train\imgs", exist_ok=True)
            cropped_mask_path = os.path.join(r"..\Pytorch-UNet-masterv2\N500x500\data_train\masks", f'crop_{mask_filename}')
            cropped_original_path = os.path.join(r"..\Pytorch-UNet-masterv2\N500x500\data_train\imgs", f'crop_{os.path.basename(sorted_OriginalsInTrack[j])}')

            # Save the cropped images
            cv2.imwrite(cropped_mask_path, resized_mask)
            cv2.imwrite(cropped_original_path, resized_image)

            print(f'Saved cropped mask to: {cropped_mask_path}')
            print(f'Saved cropped original image to: {cropped_original_path}')
