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

# Function to crop and resize the window around the object
def crop(mask, rmin, rmax, cmin, cmax):
    obj_height = rmax - rmin + 1
    obj_width = cmax - cmin + 1

    window_size = 350

   

    if obj_height > window_size or obj_width > window_size:
        # If the object is larger than 150x150, use the full bounding box and resize
        cropped = mask[rmin:rmax+1, cmin:cmax+1]
        resized = cv2.resize(cropped, (window_size, window_size))
    else:
        # If the object fits within a 150x150 window, create a centered window
        center_r = (rmin + rmax) // 2
        center_c = (cmin + cmax) // 2
        start_r = max(center_r - window_size/2, 0)
        start_c = max(center_c - window_size/2, 0)
        end_r = min(center_r + window_size/2, mask.shape[0])
        end_c = min(center_c + window_size/2, mask.shape[1])
        
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

# List of image paths
Train = glob.glob(os.path.join(r'Small_ARIS_Mauzac_UnetReady_MergedTracks\Test\2014*'))
#Val = glob.glob(os.path.join(r'C:\Users\d42684\Documents\STAGE\CODES\Small_ARIS_Mauzac\Val\2014*'))
#Test = glob.glob(os.path.join(r'C:\Users\d42684\Documents\STAGE\CODES\Small_ARIS_Mauzac\Test\2014*'))

#AllVideos = Train + Val + Test
AllVideos = Train

# Iterate over the provided image paths
for VideoPath in AllVideos:

    TrackspathsMasks = glob.glob(os.path.join(VideoPath,"Foreground","t*"))
    TrackspathsOriginals = glob.glob(os.path.join(VideoPath,"Original","t*"))

    if len(TrackspathsMasks) != len(TrackspathsOriginals):
        print("Error : Video doesnt have the same tracks for gt and input")
        break

    for i in range(len(TrackspathsMasks)):

        MasksInTrack = glob.glob(os.path.join(TrackspathsMasks[i],"m_*"))
        sorted_MasksInTrack = sorted(MasksInTrack, key=extract_frame_number)

        OriginalsInTrack = glob.glob(os.path.join(TrackspathsOriginals[i],"2014*"))
        sorted_OriginalsInTrack = sorted(OriginalsInTrack, key=extract_frame_number)

        if len(sorted_MasksInTrack) != len(sorted_OriginalsInTrack):
                print("Error : Video doesnt have the same tracks for gt and input")
                break



        #if mask_filepath.endswith(('.png', '.jpg')):
        for j in range(len(sorted_MasksInTrack)):

            

            mask_filename = os.path.basename(sorted_MasksInTrack[j])
            mask_dir = os.path.dirname(sorted_MasksInTrack[j])
            videoparts = sorted_MasksInTrack[j].split("\\")

            #UNtilVideo = os.path.join(videoparts[0],videoparts[1],videoparts[2],videoparts[3],videoparts[4],videoparts[5],videoparts[6],videoparts[7],videoparts[8],"Original")
            # Extract date and frame number from the mask filename
            mask_parts = mask_filename.split('_')

            if "Arcing" in mask_parts[2]:
                
                date_part = mask_parts[3] + '_' + mask_parts[4]
            else:
                date_part = mask_parts[2] + '_' + mask_parts[3]

            frame_part = mask_parts[-1].split('.')[0].replace('frame', '')

            # # Match with the corresponding original image
            # matched_original = None
            # for original_filepath in AllOriginalImages:
            #     original_filename = os.path.basename(original_filepath)
            #     if date_part in original_filename and frame_part in original_filename:
            #         matched_original = original_filepath
            #         break
            #     if date_part == "2014-11-08_150000":
            #         hola = input('wait')

            
            # Load the mask and original image
            mask = cv2.imread(sorted_MasksInTrack[j], cv2.IMREAD_GRAYSCALE)
            original_image = cv2.imread(sorted_OriginalsInTrack[j])

            # Get the bounding box containing all non-zero values
            rmin, rmax, cmin, cmax = get_bounding_box(mask)

            # Crop and resize the window around the object in the mask
            resized_mask = crop(mask, rmin, rmax, cmin, cmax)

            # Crop and resize the window around the object in the original image
            resized_image = crop(original_image, rmin, rmax, cmin, cmax)

            # Output paths for cropped images #"D:\Doble Titulación\Programa_ECN\ANO2\TFE\EDF\CODE\Pytorch-UNet-masterv2\data\imgs"
            #cropped_mask_path = os.path.join(mask_dir, f'crop_{mask_filename}')
            #cropped_original_path = os.path.join(os.path.dirname(sorted_OriginalsInTrack[j]), f'crop_{os.path.basename(sorted_OriginalsInTrack[j])}')
#"D:\Doble Titulación\Programa_ECN\ANO2\TFE\EDF\CODE\Pytorch-UNet-masterv2\150x150\data_test_MergedTracks"
            cropped_mask_path = os.path.join(r"..\Pytorch-UNet-masterv2\350x350\data_test\masks", f'crop_{mask_filename}')
            cropped_original_path = os.path.join(r"..\Pytorch-UNet-masterv2\350x350\data_test\imgs", f'crop_{os.path.basename(sorted_OriginalsInTrack[j])}')

            # Save the resized images with "crop_" prefix   
            cv2.imwrite(cropped_mask_path, resized_mask)
            cv2.imwrite(cropped_original_path, resized_image)

            print(f'Saved cropped mask to: {cropped_mask_path}')
            print(f'Saved cropped original image to: {cropped_original_path}')

