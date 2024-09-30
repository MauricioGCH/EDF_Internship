#main to extract BGS frames using the annotation file; this was just to extract the subsense results for the bgs input version of unet

import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import datetime
import config
import cv2
import warnings
import pandas as pd
# Filter out FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

import glob
import inspect
from skimage import io
import re
import ctypes
import sys

parent_dir = r"C:\Users\chapi\Documents\STAGE\CODE\bgslibrary"

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

ctypes.cdll.LoadLibrary("C:\\OpenCV490\\build\\x64\\vc16\\bin\\opencv_world490.dll")
import build_win64.pybgs as bgs


def main(Dataset, BGS_Method, Filter):



    for video in Dataset:
        videoname = os.path.basename(video)
        tracks = glob.glob(os.path.join(video, "Original","*"))

        date = videoname.split("_")[0]
        date = date.replace("-", "_")
        videopath = os.path.join(r'E:\Data',str('ARIS_'+ date+'_AVI'), str(videoname+'.avi'))
        t1 = datetime.datetime.now()       
                            
        path = videopath
        #print(path)
        capture = cv2.VideoCapture(path)
        print("the path is : ",path)
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.set(cv2.CAP_PROP_POS_FRAMES, 1) # Moving to the second frame
        
        fgbg = BGS_Method
        
        while True:

            for track in tracks:
                trackname = os.path.basename(track)
                #, "2014*.jpg"
                frames_pertrack_pervideo = glob.glob(os.path.join(video,"Original",trackname, "2014*.jpg"))
                os.makedirs(os.path.join(video,"BGS",trackname), exist_ok=True)
                print("rr")
                MovingBeforeTrack = 0
                for frame  in frames_pertrack_pervideo:
                    framename = os.path.basename(frame)
                    match = re.search(r'Obj_frame(\d+)', framename)
                    if match:
                        framenumber = int(match.group(1))
                    
                    #capture.set(cv2.CAP_PROP_POS_FRAMES, framenumber + 2) # Moving to the second frame       

                    
                    while True:
                        (grabbed, img_frame) = capture.read()
                        if not grabbed:
                            #print("video not loaded")
                            break
                        img_frame = cv2.cvtColor(img_frame,cv2.COLOR_RGB2GRAY)

                        if MovingBeforeTrack == 0:
                            capture.set(cv2.CAP_PROP_POS_FRAMES, framenumber - 30)
                            MovingBeforeTrack = 1
                    
                        frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))-2
                        print(frame)

                        ## img_frame
                        if Filter == "Gaussian":
                            dst = cv2.GaussianBlur(img_frame, (5, 5), 0)
                        elif Filter == "Mean":
                            kernelFiltering = np.ones((6,4),np.float64)/24
                            dst = cv2.filter2D(img_frame,-1,kernelFiltering)
                        elif Filter == "Median":
                            dst = cv2.medianBlur(img_frame, 5)


                        binaryMask = fgbg.apply(dst) ## background substraction
                        
                        if frame ==framenumber:
                            
                            # # Create a subplot with 1 row and 2 columns
                            # fig, axs = plt.subplots(1, 2, figsize=(10, 5))

                            # # Display the first image
                            # axs[0].imshow(img_frame, cmap ="gray")
                            # axs[0].set_title(framename)
                            # axs[0].axis('off')  # Hide the axis

                            # # Display the second image
                            # axs[1].imshow(binaryMask, cmap ="gray")
                            # axs[1].set_title(str("BGS_"+framename))
                            # axs[1].axis('off')  # Hide the axis

                            # # Adjust layout
                            # plt.tight_layout()

                            # # Show the plot
                            # plt.show()

                            cv2.imwrite(os.path.join(video,"BGS",trackname, str("BGS_"+framename) ), binaryMask.astype(np.uint8))
                            
                            break
                            
                        
            break



if __name__ == "__main__":

    VideoList = glob.glob(r"C:\Users\chapi\Documents\STAGE\CODE\bgslibrary\Small_ARIS_Mauzac_UnetReady_Final_BGS\*\*")
    BGS_Method = bgs.SuBSENSE()
    Filter ="Gaussian" #Mean Gaussian Median



    main(VideoList, BGS_Method, Filter)

                    



                    


    