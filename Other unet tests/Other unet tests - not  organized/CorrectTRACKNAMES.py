import os
import re

def rename_images(folder_path):
    # Function to extract frame number from filename
    def extract_frame_number(filename):
        match = re.search(r'frame(\d+)', filename)
        if match:
            return int(match.group(1))
        return None

    # Get all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # Create a dictionary to map frame numbers to filenames
    frame_to_file = {extract_frame_number(f): f for f in files if extract_frame_number(f) is not None}
    
    # Prepare new filenames (first to temporary names)
    temp_filenames = {}
    new_frame_number = 72

    # Process frames 93 to 122
    for frame_number in range(93, 123):
        if frame_number in frame_to_file:
            old_filename = frame_to_file[frame_number]
            temp_filename = re.sub(r'frame\d+', f'frame_tmp{new_frame_number}', old_filename)
            temp_filenames[old_filename] = temp_filename
            new_frame_number += 1

    # Process frames 72 to 92
    for frame_number in range(72, 93):
        if frame_number in frame_to_file:
            old_filename = frame_to_file[frame_number]
            temp_filename = re.sub(r'frame\d+', f'frame_tmp{new_frame_number}', old_filename)
            temp_filenames[old_filename] = temp_filename
            new_frame_number += 1

    # Rename files to temporary names
    for old_filename, temp_filename in temp_filenames.items():
        old_filepath = os.path.join(folder_path, old_filename)
        temp_filepath = os.path.join(folder_path, temp_filename)
        os.rename(old_filepath, temp_filepath)
        print(f'Renamed {old_filename} to {temp_filename}')

    # Prepare final filenames from temporary names
    final_filenames = {}
    for temp_filename in temp_filenames.values():
        final_filename = re.sub(r'frame_tmp', 'frame', temp_filename)
        final_filenames[temp_filename] = final_filename

    # Rename files from temporary names to final names
    for temp_filename, final_filename in final_filenames.items():
        temp_filepath = os.path.join(folder_path, temp_filename)
        final_filepath = os.path.join(folder_path, final_filename)
        os.rename(temp_filepath, final_filepath)
        print(f'Renamed {temp_filename} to {final_filename}')

# Example usage
folder_path = r'D:\Doble Titulación\Programa_ECN\ANO2\TFE\EDF\CODE\U-Net\Small_ARIS_Mauzac_UnetReady_Final\NewMasks4\2014-11-07_071000\Original\t0'
rename_images(folder_path)