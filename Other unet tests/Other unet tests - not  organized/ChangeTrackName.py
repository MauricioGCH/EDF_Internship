import os

# Define the directory containing your images
directory = r"D:\Doble Titulación\Programa_ECN\ANO2\TFE\EDF\CODE\U-Net\Small_ARIS_Mauzac_UnetReady_Final\Train\2014-11-05_184000\Foreground\t4"  # Change this to your folder path

# Loop through each file in the directory
for filename in os.listdir(directory):
    # Check if the file contains 't4'
    if 't4' in filename:
        # Replace 't4' with 't1'
        new_filename = filename.replace('t4', 't1')
        # Construct the full file paths
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)
        # Rename the file
        os.rename(old_file, new_file)
        print(f'Renamed: {filename} to {new_filename}')