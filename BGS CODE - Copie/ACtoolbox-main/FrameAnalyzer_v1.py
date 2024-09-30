import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import datetime
import config
import cv2

def main():
    # Initialize video capture C:\Users\d42684\Documents\STAGE\CODES\ACtoolbox-main\Dataset\Labels\Comptage2014-11-05_1906_vf.xlsx
    #cap = cv2.VideoCapture(os.path.join(config.pathAVI, config.videoName + '.avi'))  # Replace with your video file path
    Hardrive = True
    if Hardrive :
        #F:\Sélune_ARIS\DonneesARIS_Azenor\SIL
        #Specific_Dataset = 'ARIS_Mauzac'
        #HardriveDay = 'ARIS_2014_11_05_AVI'
        videoName = '2019-05-02_005000.avi' #1358 last reviewed excel line

        
        #path = os.path.join('E:\Data',HardriveDay,videoName +'.avi')
        #print(path)
        #cap = cv2.VideoCapture(path)
        #cv2.imwrite(os.path.join("Dataset",Small_ARIS_SELUNE,videoName,FP_FN_str,group_folder,filename), frame)
        cap = cv2.VideoCapture(os.path.join(r"F:\Sélune_ARIS\DonneesARIS_Azenor\SIL",videoName))
    else :
        Specific_Dataset = config.Specific_Dataset
        videoName = config.videoName

        cap = cv2.VideoCapture(os.path.join(config.pathAVI, videoName + '.avi'))

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('fps : ',cap.get(cv2.CAP_PROP_FPS) )
    
    print('total frames ', total_frames)
    current_frame = 0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #group_name = 1  # Initial group name
    os.makedirs(os.path.join("Dataset","Small_ARIS_SELUNE",videoName,"False Negatives"), exist_ok=True)
    os.makedirs(os.path.join("Dataset","Small_ARIS_SELUNE",videoName,"False Positives"), exist_ok=True)
    
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Frame', frame_width, frame_height)

    while True:
        # Set the video frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        
        # Read a frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break


        group_name = 1 # For saving tracks
        # Display the frame
        cv2.imshow('Frame', frame)

        # Wait for key press
        key = cv2.waitKey(0) & 0xFF
        #print(key)
        # Navigation controls
        if key == ord('q'):  # Quit
            break
        
        elif key == 54:  # Next frame

            current_frame += 1
            print("Current frame is : " + str(current_frame))
            if current_frame >= total_frames:
                current_frame = 0
        
        elif key == 52:  # Previous frame

            current_frame -= 1
            print("Current frame is : " + str(current_frame))
            if current_frame < 0:
                current_frame = total_frames - 1

        elif key == ord('g'):  # Go to specific frame

            frame_num = input("Enter frame number (0 to {}): ".format(total_frames - 1))
            current_frame = int(frame_num)
            if current_frame >= total_frames or current_frame < 0:
                print("Invalid frame number.")
                current_frame = 0

            print("Current frame is : " + str(current_frame))

        elif key == ord('s'):  # Save frames
            
            FP_FN = input(" Recording FP(1) or FN(2) ? ")
            if int(FP_FN) == 1:
                FP_FN_str = "False Positives"
            elif int(FP_FN) == 2 :
                FP_FN_str = "False Negatives"
            ObjectName = input(" what is the Object to track (remember to name it according format : eel (when FN or TP), otherfish(FP), débris(FP), arcing(FP), doubletrack(FP)) ? ")

            start_frame = int(input("Select starting frame : "))
            end_frame = int(input("Select end frame : "))

            group_folder = f"{ObjectName}_{group_name}"
            
            while True:
                if os.path.exists(os.path.join("Dataset","Small_ARIS_SELUNE",videoName,FP_FN_str,group_folder)):
                    group_name+=1
                    group_folder = f"{ObjectName}_{group_name}"
                    continue
                else:
                # Create directory
                    os.mkdir(os.path.join("Dataset","Small_ARIS_SELUNE",videoName,FP_FN_str,group_folder))
                    break
            

            for frame_num in range(start_frame, end_frame + 1):
                
                current_frame = frame_num ##

                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame) ##

                ret, frame = cap.read()

                if not ret:
                    print("Error: Could not read frame.")
                    break
                
                filename = f"{ObjectName}_{frame_num}.png"

                cv2.imwrite(os.path.join("Dataset","Small_ARIS_SELUNE",videoName,FP_FN_str,group_folder,filename), frame)
                
                
            
            print(f"Finished recording of previous track {group_name}. (frame {start_frame} to  frame {end_frame})")
            #group_name += 1
            #print(f"Increasing track number to {group_name} for a new recording.")


    # Release the video capture and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()