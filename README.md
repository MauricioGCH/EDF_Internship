# Repository Overview

This repository contains 6 main folders, which I will explain below:

## 1. BGS CODE
This folder contains the code based on Azenor's work, implementing the Kalman filter along with the BGS method. Currently, the BGS method can be modified (it is crucial to correctly install the PyBGS library). The following `.py` files are found within this folder:
- **Extract_past_frames.py**: Simply extracts the previous frames from a given track.
- **BGSFrames_Extraction.py**: Created to extract raw SuBSENSE results for the addition of the UNET method with the extra BGS channel.
- **FrameAnalyzer_v1.py**: A code designed to navigate through videos. It allows advancing and rewinding frame by frame, jumping to a specific frame, and selecting a specific segment to save with a specific name according to the available options. This was essential for qualitative analysis.
- **Labels_Organizer.py**: Initially created to organize the manual annotations from Azenor's work (it created the `Organized_Labels` file, which was widely used to easily navigate the videos).
- **Masks_Overlay.py**: Used for visualizing results.
- **Properties_datasets.py**: Used for video inventory.
- **Main.py**: The main file for generating prompts in a format readable by SAM.
- The `src` folder contains important secondary functions.
- Some files are called `SecondVersion_Annotations_Filled`, but the final versions of all the data will be in a separate folder, so you can ignore these in this folder.

## 2. Other UNET Tests
This contains both an organized and unorganized version. This folder generally contains functions used to modify the database once the final SAM annotations were available. The most important are `Save_Cropv5.py` and `Save_Cropv6.py`, which created the cropped versions of the database. `v6` was modified to save images by track instead of all together to enable track-by-track analysis of results. To understand the differences between the two, you need to carefully examine how the files are named and the folder creation order for the images.

## 3. MAIN UNET Not Organized
This folder contains the code I used for training the UNET model on the EDF cluster. It is an unorganized version that contains `.py` files from various attempts to optimize hyperparameters with different libraries and older training and evaluation functions.

## 4. MAIN UNET Organized
The organized version of the main UNET training code:
- **OptunaOptim.py**: Training code for automatic hyperparameter optimization of the UNET model.
- **Visualization_test.py**: Generates predictions from a trained model for qualitative comparison.
- **Evaluate3.py**: Contains model evaluation functions and secondary visualization functions.
- **test1.py**: Used for model evaluation, mainly on the test split. It calculates Jaccard, confusion matrices, and classification reports per class.
- **TrackPrediction.py**: For track-by-track analysis of results. It generates a BEFORE and AFTER version of model predictions on the track based on the majority class. Changes are also saved in a `.txt` file per track.
- **Train_noWandb_uniquefolder**: The MAIN training code for the model. The name isn’t very intuitive, as it was more indicative for personal organization and to avoid confusion with other versions.
- The `unet` folder contains the UNET model structure code.
- The `utils` folder contains secondary functions for training the UNET model.
- The `SH Batch Files` folder contains all batch files for running the code on the EDF cluster.

## 5. segment-anything-main – not clean
This is the unclean version of the code used for sending the annotations obtained from BGS CODE in the correct format for SAM. It contains older versions used for experimenting with modifications to the prompts without changing the main versions.

## 6. segment-anything-main – Cleaned
The clean version of the SAM code for generating our segmentation masks for the database. It contains the important elements inside the `notebooks` folder:
- **PromptsVIA_to_SAM_SecondVersion FINAL.ipynb**: The code that loads the prompts obtained from BGS CODE and sends them correctly to the SAM model, organizing the results.
- **Mask_visualization.py**: A function used at one point to visualize results.
- **SELUNE.csv**: A track annotated to experiment with adding data from the Selune.

The database will first be stored on the desktop computer in my EDF office, on the SELUNE ARIS hard drive, and I will also send you a link to download it. The database folder contains:
- **Organized_Labels**: The Excel file with Azenor's manually organized annotations.
- **SecondVersion_Annotations_Filled.csv**: The final version of bounding box and medoid prompts from BGS_CODE to send to SAM.
- **Small_ARIS_Mauzac_UnetReady_Final_BGS**: The original and final version of the database, with other adapted versions based on the model to be trained or evaluated. There is likely a more optimal way to handle these different versions without copying the entire database, which would be especially important as the database grows, but for now, it’s organized this way.
- The folders named after resolutions (e.g., `350x350`, `500x500`) are used for both the normal UNET and the UNET with an additional BGS channel.
- The **FullSize1276x664** folder is specifically for the 3-frame model due to the special cropping process required.
- Folders with a `T` at the beginning are used for track-by-track performance analysis of the model. The only difference is that the images are organized by track rather than being grouped without distinction. This only works for evaluating the UNET model without temporal information and with BGS information.
- Folders with a `D` at the beginning serve the same purpose as the `T` folders, but for the 3-frame version with temporal information.

There is also a folder called LastExpsOptuna

As you can see, the database management can be somewhat confusing if you're not familiar with it, as it’s not optimally organized. A good task would be to better organize how the UNET dataset is defined for its 3 versions to simplify database management.
