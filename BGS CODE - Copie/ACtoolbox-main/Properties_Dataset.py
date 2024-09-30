## Properties from dataset
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import datetime
import config
import cv2
import glob
import pandas as pd


#r"F:\Sélune_ARIS\DonneesARIS_Azenor\Carpe"
#file_full_path = glob.glob(os.path.join(full_path,"*xlsx"))

All_folders = glob.glob(os.path.join(r"F:\Sélune_ARIS\DonneesARIS_Selune","*"))
#path = os.path.join('E:\Data',HardriveDay,videoName +'.avi')
#print(path)

Dataset_frames = 0
Average_fps = 0
Video_quantity = 0
Dataset_size = 0
print(All_folders)
for folder in All_folders :
    folder_videos = glob.glob(os.path.join(folder,'*.avi'))
    #print(folder_videos)
    for video in folder_videos:

        cap = cv2.VideoCapture(video)

        Dataset_frames += cap.get(cv2.CAP_PROP_FRAME_COUNT)
        Average_fps += cap.get(cv2.CAP_PROP_FPS) # division at the end
        Video_quantity += 1
        Dataset_size += os.path.getsize(video)
        #print("wait")
Average_fps = Average_fps/Video_quantity

#labels = os.path.join('Dataset', 'Labels','Organized_Labels.xlsx')
#df = pd.read_excel(labels) 
#eelCount = df['GT_Count'].sum()
#eelFrames = eelCount*2*Average_fps
#Past_FP = df['OldFP_Count'].sum()

print(f"Total Number of Videos: {Video_quantity}")
print(f"Total Number of Frames: {int(Dataset_frames)}")

print(f"Average FPS across all videos: {Average_fps:.2f}")
print(f"Total Duration of all videos: {(Dataset_frames/Average_fps)/360:.2f} Hours")
print(f"Total Size of all videos: {Dataset_size/(1024**3):.2f} GB")
#print(f"Total eel count : {eelCount}")
#print(f"Considering that each eel is at least recorded 2 seconds, the total number of frames with eels is : {eelFrames}")
#print(f"Total FP from old method (Useful as hard examples) : {Past_FP}")



""" def extract_date(date_time_str):
    return date_time_str.split('_')[0]

# Apply function to extract date components
df['VideoName'] = df['VideoName'].apply(extract_date)

# Convert 'date_column' to datetime format
df['VideoName'] = pd.to_datetime(df['VideoName'], format = '%Y-%m-%d') # '%Y-%m-%d_%H%M%S'

# Set 'date_column' as the index of the DataFrame
df.set_index('VideoName', inplace=True)

df.sort_index(inplace=True)

sum_by_date = df['GT_Count'].groupby(df.index).sum()

# Drop duplicate dates
sum_by_date = sum_by_date[~sum_by_date.index.duplicated()]

# Plot the time series
plt.figure(figsize=(10, 6))
plt.bar(sum_by_date.index, sum_by_date)
plt.title('Eel Count per Day')
plt.xlabel('Date')
plt.ylabel('Eels')
plt.grid(True)
plt.tight_layout()

plt.show() """