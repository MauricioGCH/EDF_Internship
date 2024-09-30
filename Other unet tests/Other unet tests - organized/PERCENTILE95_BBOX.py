import os
import glob
import cv2
import numpy as np
from scipy.stats import scoreatpercentile

# Function to extract bounding boxes from a mask
def extract_bounding_box(mask_path):
    # Load the mask image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize bounding box coordinates
    x_min, y_min, x_max, y_max = np.inf, np.inf, -np.inf, -np.inf
    
    # Iterate over contours to find the smallest bounding box encompassing all contours
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    
    # Compute width and height of the bounding box
    width = x_max - x_min
    height = y_max - y_min
    
    return width, height

# Function to process all masks in a video path
def process_video_path(video_path):
    mask_paths = glob.glob(os.path.join(video_path, "Foreground", "t*", "m_*"))
    widths, heights = [], []
    
    for mask_path in mask_paths:
        width, height = extract_bounding_box(mask_path)
        widths.append(width)
        heights.append(height)
    
    return widths, heights

# Function to process all videos in train, val, and test sets
def process_all_videos(video_paths_pattern):
    video_paths = glob.glob(video_paths_pattern)
    all_widths, all_heights = [], []
    
    for video_path in video_paths:
        widths, heights = process_video_path(video_path)
        all_widths.extend(widths)
        all_heights.extend(heights)
    
    return all_widths, all_heights

# Calculate the 95th percentile for a list of values
def calculate_percentile(values, percentile):
    return scoreatpercentile(values, percentile)

# Main function to process train, val, and test sets and compute 95th percentiles
def main():
    train_videos_pattern = r"Small_ARIS_Mauzac_UnetReady_Final\Train\2014*"
    val_videos_pattern = r"Small_ARIS_Mauzac_UnetReady_Final\Val\2014*"
    #test_videos_pattern = r"Small_ARIS_Mauzac_UnetReady_Final\Test\2014*"
    
    all_widths, all_heights = [], []
    
    for video_pattern in [train_videos_pattern, val_videos_pattern]:#, test_videos_pattern
        widths, heights = process_all_videos(video_pattern)
        all_widths.extend(widths)
        all_heights.extend(heights)
    
    width_95th_percentile = calculate_percentile(all_widths, 100)
    height_95th_percentile = calculate_percentile(all_heights, 100)
    
    print(f"95th Percentile Width: {width_95th_percentile}")
    print(f"95th Percentile Height: {height_95th_percentile}")

# Run the main function
if __name__ == "__main__":
    main()