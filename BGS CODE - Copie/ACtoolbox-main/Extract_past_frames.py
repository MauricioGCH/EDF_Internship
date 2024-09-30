## NEW SCRIPT TO ADD PAST FRAMES TO TRACKS? OS THAT WE CAN MODELIZE THE BACKGROUND

import os
import cv2
import glob

def extract_frames(video_path, track_folder):
    # Get list of objects (Obj_frame#) in the track folder

    # Sort the list of objects numerically

    lowest_frame = int(os.listdir(track_folder)[0].split(".")[0].split("_")[1].split("e")[1])
    # Get the lowest frame number

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Calculate the starting frame number
    start_frame = max(0, lowest_frame - 100)

    # Extract the 100 past frames
    for i in range(start_frame, lowest_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            # Save the frame as a .jpg file in the track folder
            cv2.imwrite(os.path.join(track_folder, f"Obj_frame{i}.jpg"), frame)

    # Release the video capture object
    cap.release()

# Example usage
video_path = "E:\\Data\\ARIS_2014_11_12_AVI\\2014-11-12_201000.avi"
track_folder = "Test\\ARIS_Mauzac\\2014-11-12_201000\\Track_final_Images\\Original\\*"  # Example track folder

list_tracks = glob.glob(track_folder)

for track in list_tracks:

    extract_frames(video_path, track)
