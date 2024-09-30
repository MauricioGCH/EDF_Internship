
# Define the folder containing the images
import os
import re
import glob
import os
import re
import glob

# Define the path to search for files
division_paths = glob.glob(r"D:\Doble Titulación\Programa_ECN\ANO2\TFE\EDF\CODE\U-Net\Small_ARIS_Mauzac_UnetReady_MergedTracks\*\*\Foreground\*\m_*")

for file_path in division_paths:
    # Get the directory and filename from the file_path
    folder_path, filename = os.path.split(file_path)

    # Define the pattern to match and remove the number after "m_*"
    pattern = re.compile(r'(m_[A-Za-z]+(?:_[A-Za-z]+)*_?)(\d+_)(.*)')
    
    # Check if the filename matches the pattern
    match = pattern.match(filename)
    if match:
        # Extract the parts of the filename
        part1 = match.group(1)
        part2 = match.group(3)
        
        if part2[0] == "_":
            # Create the new filename
            new_filename = f"{part1}{part2}"
        else:
            new_filename = f"{part1}_{part2}"

        
        # Get the full paths
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_filename)
        
        # Rename the file
        os.rename(old_file, new_file)
        print(f"Renamed: {filename} -> {new_filename}")
    else:
        # If no match, the filename is already correct
        print(f"Unchanged: {filename}")

