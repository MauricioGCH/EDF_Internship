



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

from src.calibration_functions import func_ratio_px
from src.detection_functions import select_candidate_img,detectROI,img_reconstruction
from src.morphology_functions import calculation_candidate_morphology
from src.tracking_functions import Track
from src.utils import random_color
import glob



import ctypes


import sys

parent_dir = r"C:\Users\chapi\Documents\STAGE\CODE\bgslibrary"

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

ctypes.cdll.LoadLibrary("C:\\OpenCV490\\build\\x64\\vc16\\bin\\opencv_world490.dll")
import build_win64.pybgs as bgs

def main(videopath, Specific_Dataset, VideoBool, BGS_Method, ImagesList, Filter):

    t1 = datetime.datetime.now()
    
    fish = []
    if VideoBool:

        Hardrive = True
        if Hardrive :           
            
            path = videopath
            #print(path)
            capture = cv2.VideoCapture(path)
            print("the path is : ",path)
        else :
            #Specific_Dataset = config.Specific_Dataset

            capture = cv2.VideoCapture(os.path.join(config.pathAVI, config.videoName + '.avi'))


        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.set(cv2.CAP_PROP_POS_FRAMES, 1) # Moving to the second frame
    else :
        fps = 7 #default in aRIs mauzac, to change later
    

    
    
    
    track_in_process, track_final = [],[] ## They contain Track class elements

    #fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=config.seuil_mean,history=int(14*fps),detectShadows = False)
    fgbg = BGS_Method

    kernelFiltering = np.ones((6,4),np.float64)/24 
    
    affichage = False
    affichage2 = True
    
    ##------------------------------------
    ## Setting up Relative paths
    absolute_path = os.path.dirname(__file__)

    relative_path = 'Test'

    if not os.path.exists(relative_path):
        os.makedirs(relative_path)
        print(f"Folder '{relative_path}' created successfully.")
    else:
        print(f"Folder '{relative_path}' already exists.")

    
    full_path = os.path.join(absolute_path, relative_path)
    print(full_path)
    os.makedirs(full_path, exist_ok= True)
    ##------------------------------------

    ##--------------------
    
    if affichage:
        frame_width = 664
        frame_height = 1276
        outTracking = cv2.VideoWriter(os.path.join(full_path,'Tracking_test.avi'),cv2.VideoWriter_fourcc('M','J','P','G'),fps,(frame_width,frame_height))
        outBinary = cv2.VideoWriter(os.path.join(full_path,'Binary_test.avi'),cv2.VideoWriter_fourcc('M','J','P','G'),fps,(frame_width,frame_height))
        outExplication= cv2.VideoWriter(os.path.join(full_path,'Explication_test.avi'),cv2.VideoWriter_fourcc('M','J','P','G'),fps,(frame_width + 500, frame_height + 200))
    if not VideoBool:
        NumberImages = len(ImagesList)
    ImgIterator = 0 # to change image and also serves as frame counter
    while True: # Start of frame-by-frame analysis for frame in range(nb_frames+1):#


        if VideoBool: # loading video frames or images from image sequence, the break is to stop when sequence is over
            (grabbed, img_frame) = capture.read()
            NumberFirstImage = 0
            if not grabbed:
                #print("video not loaded")
                break
                
        else:
            if NumberImages > ImgIterator:
                img_frame = cv2.imread(ImagesList[ImgIterator])

                frame_width = img_frame.shape[1]
                frame_height = img_frame.shape[0]


                NumberFirstImage = int(ImagesList[0].split("\\")[-1].split("_")[4].split(".")[0][5:])

            else:
                break


        img_display = img_frame.copy()
        img_display2 = img_frame.copy()
        
        img_frame = cv2.cvtColor(img_frame,cv2.COLOR_RGB2GRAY)
        

        if VideoBool:
            frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))-2

            if frame == 0 :
                aire_px,x_max,range_manquant,y_mil,long_px = func_ratio_px(img_frame,config.rangeCameraMin,config.rangeCameraMax)
                
                fps = int(np.round(capture.get(cv2.CAP_PROP_FPS)))


        else:
            frame = ImgIterator

            if frame == 0 :
                aire_px,x_max,range_manquant,y_mil,long_px = func_ratio_px(img_frame,config.rangeCameraMin,config.rangeCameraMax)
                
                

        
        

        
        
        
        # Remove track in process when frame > epsi_frame
        ind_remove = []
        # Update tracks: handles the lifecycle of tracks in progress
        """focuses on updating tracks in progress (track_in_process) and managing their creation, update, and 
        removal based on certain conditions, such as the distance from the last frame and the current frame number. """

        for tr in range(len(track_in_process)): 
            if frame> track_in_process[tr].last_frame +config.epsi_frame: # Tracking is ended
                if len(track_in_process[tr].tot_frame) > 3:
                    
                    candidate,result = select_candidate_img(track_in_process[tr])
                    if result==True:
                        track_final.append(track_in_process[tr])
                        candidate = calculation_candidate_morphology(candidate)
                        candidate.redress_image()
                    
                        fish.append(candidate)
                         
                        
                        
                ind_remove.append(tr)
            else:
                track_in_process[tr].predict(config.A)
        track_in_process=list(np.delete(track_in_process,ind_remove))


        ## img_frame
        if Filter == "Gaussian":
            dst = cv2.GaussianBlur(img_frame, (5, 5), 0)
        elif Filter == "Mean":
            dst = cv2.filter2D(img_frame,-1,kernelFiltering)
        elif Filter == "Median":
            dst = cv2.medianBlur(img_frame, 5)
        
        #dst = cv2.filter2D(img_frame,-1,kernelFiltering) ## In the paper didn't it say it did a median filter to remove pepper salt noise ?
        #dst = cv2.GaussianBlur(img_frame, (5, 5), 0)
        #dst = cv2.medianBlur(img_frame, 3)
        binaryMask = fgbg.apply(dst) ## background substraction 
        
        binaryMaskTest = cv2.cvtColor(binaryMask,cv2.COLOR_GRAY2RGB)

        if isinstance(fgbg, cv2.BackgroundSubtractorMOG2):
            Background_Estimation = fgbg.getBackgroundImage()
        else:
            Background_Estimation = fgbg.getBackgroundModel()
        

        if affichage:
            imgExplanation_to_write = np.zeros((frame_height + 200, frame_width + 500, 3), dtype=np.uint8)
            imgExplanation_to_write[:img_display.shape[0],:img_display.shape[1],:]=img_display
            imgExplanation_to_write = cv2.putText(imgExplanation_to_write, 'Frame #'+str(frame), (frame_width, 60) , cv2.FONT_HERSHEY_SIMPLEX ,2, (0, 0, 255) , 5, cv2.LINE_AA) 
            outExplication.write(imgExplanation_to_write.astype(np.uint8))

        if frame>10:#frame>1740 and frame <1768:#frame>500 and frame<900:#frame>1740 and frame <1768:#frame >500:# and frame>1740 and frame <1768:
            
            isInterest,regionsCandidate = detectROI(binaryMask,config.long/long_px,long_px)  
            count = 0
    
            if regionsCandidate:
                for reg in regionsCandidate :
                    img_crop = img_frame[reg.bbox[0]:reg.bbox[2],reg.bbox[1]:reg.bbox[3]]
                    img_crop_binary = binaryMask[reg.bbox[0]:reg.bbox[2],reg.bbox[1]:reg.bbox[3]]
                    
                    
                    
                    count+=1
                    img_to_reconstruct_binary = np.pad(img_crop_binary,((100,100),(100,100)), constant_values=((0,0),(0,0)))
                    img_to_reconstruct = np.pad(img_crop,((100,100),(100,100)), constant_values=((0,0),(0,0)))
                    img_reconstructed,length = img_reconstruction(img_to_reconstruct_binary,reg.orientation)
                    
                    
                    
                    # Tracking
                    dis_to_track = 1000*np.ones(len(track_in_process))  
                    # If tracks are already "opened", checked wether the new object corresponds to one of them
                    
                    if track_in_process: 
                        # Calculation of the distance to tracks in process
                        for tr in range(len(track_in_process)):
                            if track_in_process[tr].last_frame != frame:
                                if len(track_in_process[tr].tot_frame)>=2: # Use Kalman prediction 
                                    dis_to_track[tr] = np.linalg.norm(np.array([track_in_process[tr].Xp[-1][0][0],track_in_process[tr].Xp[-1][1][0]])-np.array([reg.centroid[1],reg.centroid[0]]))
                                else: # Use search area
                                    if np.abs(track_in_process[tr].orientation[-1]-reg.orientation) < config.epsi_orientation:
                                        dis_to_track[tr] = np.linalg.norm(np.array([track_in_process[tr].centroids[-1][0],track_in_process[tr].centroids[-1][1]])-np.array([reg.centroid[1],reg.centroid[0]]))/2
                        # Choose of the right track if criterion met
                        if np.sum(dis_to_track)!=1000*len(track_in_process): ## I don't understand this condition
                            bear = [int(track_in_process[np.argmin(dis_to_track)].Xp[-1][1]),int(track_in_process[np.argmin(dis_to_track)].Xp[-1][0])]
                            
                             
                            bear = [int(track_in_process[np.argmin(dis_to_track)].Xp[-1][1]),int(track_in_process[np.argmin(dis_to_track)].Xp[-1][0])]
                            if bear in reg.coords:#np.min(dis_to_track)<=150:
                                ind = np.argmin(dis_to_track)
                                track_in_process[ind].correct_and_add([reg.centroid[1],reg.centroid[0]],config.A,config.Q,config.H,config.R,frame,reg.orientation,length,img_reconstructed,img_to_reconstruct_binary,img_to_reconstruct,reg.bbox, img_display2, binaryMask, Background_Estimation)
                            else: # If not corresponding track found, creation of a new one
                                track_in_process.append(Track((reg.centroid[1],reg.centroid[0]),config.Q,frame, config.deltaT,length,img_reconstructed,fps,img_to_reconstruct_binary,img_to_reconstruct,reg.bbox, img_display2,binaryMask,Background_Estimation))
                                track_in_process[-1].predict(config.A)
                                track_in_process[-1].correct_and_add([reg.centroid[1],reg.centroid[0]],config.A,config.Q,config.H,config.R,frame,reg.orientation,length,img_reconstructed,img_to_reconstruct_binary,img_to_reconstruct,reg.bbox, img_display2,binaryMask, Background_Estimation)
                        else: # If no track started, creation of a new one
                            track_in_process.append(Track((reg.centroid[1],reg.centroid[0]),config.Q,frame, config.deltaT,length,img_reconstructed,fps,img_to_reconstruct_binary,img_to_reconstruct,reg.bbox, img_display2,binaryMask, Background_Estimation))
                            track_in_process[-1].predict(config.A)
                            track_in_process[-1].correct_and_add([reg.centroid[1],reg.centroid[0]],config.A,config.Q,config.H,config.R,frame,reg.orientation,length,img_reconstructed,img_to_reconstruct_binary,img_to_reconstruct,reg.bbox, img_display2,binaryMask, Background_Estimation)
                    # Else, starting a tracking with this new position 
                    else: 
                        track_in_process.append(Track((reg.centroid[1],reg.centroid[0]),config.Q,frame, config.deltaT,length,img_reconstructed,fps,img_to_reconstruct_binary,img_to_reconstruct,reg.bbox, img_display2,binaryMask, Background_Estimation))
                        track_in_process[-1].predict(config.A)
                        track_in_process[-1].correct_and_add([reg.centroid[1],reg.centroid[0]],config.A,config.Q,config.H,config.R,frame,reg.orientation,length,img_reconstructed,img_to_reconstruct_binary,img_to_reconstruct,reg.bbox, img_display2, binaryMask, Background_Estimation)
                    # plt.scatter(reg.centroid[1],reg.centroid[0])
            
            if affichage:
                color_bar = [(0,0,255),(255,0,0),(0,255,0),(0,255,255),(255,255,0),(255,127,0),(0,127,255),(0,255,127),(127,255,255),(255,255,127)]
                
                for tr in range(len(track_in_process)):
                    if track_in_process[tr].last_frame==frame:
                        cv2.circle(img_display,(int(track_in_process[tr].centroids[-1][0]),int(track_in_process[tr].centroids[-1][1])),5,color_bar[tr],-1)
                        cv2.circle(binaryMaskTest,(int(track_in_process[tr].centroids[-1][0]),int(track_in_process[tr].centroids[-1][1])),5,color_bar[tr],-1)
                        cv2.rectangle(img_display,(track_in_process[tr].bbox[-1][1],track_in_process[tr].bbox[-1][0]),(track_in_process[tr].bbox[-1][3],track_in_process[tr].bbox[-1][2]),color_bar[tr],3)
                        cv2.rectangle(binaryMaskTest,(track_in_process[tr].bbox[-1][1],track_in_process[tr].bbox[-1][0]),(track_in_process[tr].bbox[-1][3],track_in_process[tr].bbox[-1][2]),color_bar[tr],3)
                        cv2.circle(imgExplanation_to_write,(int(track_in_process[tr].centroids[-1][0]),int(track_in_process[tr].centroids[-1][1])),5,color_bar[tr],-1)
                        cv2.rectangle(imgExplanation_to_write,(track_in_process[tr].bbox[-1][1],track_in_process[tr].bbox[-1][0]),(track_in_process[tr].bbox[-1][3],track_in_process[tr].bbox[-1][2]),color_bar[tr],3)
                
                for tr in range(len(track_in_process)):
                    if track_in_process[tr].last_frame==frame:
        
                        img = track_in_process[tr].image[-1]
                        if img.shape[0]<300:
                            img = np.pad(img,((300-img.shape[0],0),(0,0)), constant_values=((0,0),(0,0)))
                        if img.shape[1]<300:
                            img = np.pad(img,((0,300-img.shape[1]),(0,0)), constant_values=((0,0),(0,0)))
                        if img.shape[1]>200:
                            a = int((img.shape[1]-200)/2)
                            img = img[:,a:200+a]
                        if img.shape[0]>200:
                            a = int((img.shape[0]-200)/2)
                            img = img[a:a+200,:]
                        img = img.astype(np.uint8)
                        img = cv2.resize(img, (400,400), interpolation= cv2.INTER_LINEAR)
                        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
                        imgExplanation_to_write[(frame_height - 400):frame_height,(imgExplanation_to_write.shape[1]-400):,:]=img 
                        
                        img = track_in_process[tr].img_to_reconstruct_binary[-1]
                        if img.shape[0]<300:
                            img = np.pad(img,((300-img.shape[0],0),(0,0)), constant_values=((0,0),(0,0)))
                        if img.shape[1]<300:
                            img = np.pad(img,((0,300-img.shape[1]),(0,0)), constant_values=((0,0),(0,0)))
                        if img.shape[1]>200:
                            a = int((img.shape[1]-200)/2)
                            img = img[:,a:200+a]
                        if img.shape[0]>200:
                            a = int((img.shape[0]-200)/2)
                            img = img[a:a+200,:]
                        img = img.astype(np.uint8)
                        img = cv2.resize(img, (400,400), interpolation= cv2.INTER_LINEAR)
                        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
                        imgExplanation_to_write[438:838,(imgExplanation_to_write.shape[1]-400):,:]=img
            
                        img = track_in_process[tr].img_to_reconstruct[-1]
                        if img.shape[0]<300:
                            img = np.pad(img,((300-img.shape[0],0),(0,0)), constant_values=((0,0),(0,0)))
                        if img.shape[1]<300:
                            img = np.pad(img,((0,300-img.shape[1]),(0,0)), constant_values=((0,0),(0,0)))
                        if img.shape[1]>200:
                            a = int((img.shape[1]-200)/2)
                            img = img[:,a:200+a]
                        if img.shape[0]>200:
                            a = int((img.shape[0]-200)/2)
                            img = img[a:a+200,:]
                        img = img.astype(np.uint8)
                        img = cv2.resize(img, (400,400), interpolation= cv2.INTER_LINEAR)
                        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            
                        imgExplanation_to_write[38:438,(imgExplanation_to_write.shape[1]-400):,:]=img
                        
                        outExplication.write(imgExplanation_to_write.astype(np.uint8))
                
                
        if affichage :
            outTracking.write(img_display)
            outBinary.write(binaryMaskTest) 

        if not VideoBool:

            ImgIterator = ImgIterator+ 1  

   
        
    
    if affichage :
        outTracking.release()    
        outBinary.release()
        outExplication.release()

            
    ## deals with processing completed tracks for further analysis
    for tr in range(len(track_in_process)):
        if len(track_in_process[tr].tot_frame)>3: # must have a minimum of 3 frames
            track_final.append(track_in_process[tr])
            try:
                candidate,result = select_candidate_img(track_final[-1])
                if result:
                    candidate = calculation_candidate_morphology(candidate)
                    candidate.redress_image()
                    
                    fish.append(candidate)
                
            except:
                print('Not a fish candidate in track') 

    if affichage2 :
        ## SAVING EACH IMAGE INDIVIDUALLY, IN SEPARQTES FOLDERS, THEY ARE ALSO DIVIDED INTO THE IMAGES FROM TRACK_final AND candidates (saved in fish) 
        color_bar = [(0,0,255),(255,0,0),(0,255,0),(0,255,255),(255,255,0),(255,127,0),(0,127,255),(0,255,127),(127,255,255),(255,255,127),(127,127,0),(127,0,127),(0,127,127)]
                
        bgs_used = str(BGS_Method).split(".")[-1].split(" ")[0]
        print(bgs_used)
        print("holaaaa")
        # Create folder with name from config.Specific_dataset inside "Test"
        if VideoBool:
            outputfolder = path.split("\\")[-1].split(".")[0]
            outputfolder = os.path.join(bgs_used,'VideoModeResults',outputfolder)
        else:

            xx= ImagesList[0].split("\\")[-1].split("_")

            outputfolder = os.path.join(bgs_used,"ImageModeResults",str(xx[0]+"_"+xx[1]+"_"+xx[2]+"_"+Filter))
            
        
        os.makedirs(os.path.join("Test", Specific_Dataset), exist_ok=True)
        dataset_folder = os.path.join("Test", Specific_Dataset, outputfolder)
        os.makedirs(dataset_folder, exist_ok=True)

        # Create "Track_final_Images" and "Fish_Images" folders inside the dataset folder
        track_final_images_folder = os.path.join(dataset_folder, "Track_final_Images")
        fish_images_folder = os.path.join(dataset_folder, "Fish_Images")

        os.makedirs(track_final_images_folder, exist_ok=True)
        os.makedirs(fish_images_folder, exist_ok=True)

        os.makedirs(os.path.join(track_final_images_folder,"Original"), exist_ok=True)
        os.makedirs(os.path.join(track_final_images_folder,"Foreground"), exist_ok=True)
        os.makedirs(os.path.join(track_final_images_folder,"Background"), exist_ok=True)

        os.makedirs(os.path.join(fish_images_folder,"Original"), exist_ok=True)
        os.makedirs(os.path.join(fish_images_folder,"Foreground"), exist_ok=True)
        os.makedirs(os.path.join(fish_images_folder,"Background"), exist_ok=True)
        

        headers = ['Frame','bbox','centroid']
        bbox_df = pd.DataFrame(columns=headers)

       

        # For Track_final_Images
        for idx in range(len(track_final)):
            track_folder_Original = os.path.join(track_final_images_folder,"Original" ,f"track{idx}")
            track_folder_Foreground = os.path.join(track_final_images_folder,"Foreground" ,f"track{idx}")
            track_folder_Background = os.path.join(track_final_images_folder,"Background" ,f"track{idx}")

            # for each track
            #color for this track
            color = random_color()
            
            

            os.makedirs(track_folder_Original, exist_ok=True)
            os.makedirs(track_folder_Foreground, exist_ok=True)
            os.makedirs(track_folder_Background, exist_ok=True)
            


            outBinary_tracking = cv2.VideoWriter(os.path.join(track_folder_Foreground,"Semi-automatic-Annotation.avi"),cv2.VideoWriter_fourcc('M','J','P','G'),fps,(frame_width,frame_height))

            for i in range(len(track_final[idx].img_original)):
                if i == 0 : ## Temporal fix, THE FIRST FRAME IN ALL TRACKS IS ADDED TWO TIMES AT THE BEGINNING, NEED TO FIND WHERE
                    continue
                else:
                    img = track_final[idx].img_original[i]
                    img_binary = track_final[idx].img_Complete_binary[i]
                        
                    
                    Forvideo = img_binary.copy()
                    Forvideo = cv2.cvtColor(Forvideo, cv2.COLOR_GRAY2BGR)

                    img_Background = track_final[idx].img_Background_Estimation[i]

                    bbox = track_final[idx].bbox[i]
                    if len(track_final[idx].centroids)> i:
                        centroid = (int(track_final[idx].centroids[i][0]),int(track_final[idx].centroids[i][1]))
                        cv2.circle(Forvideo,centroid,5,color,-1)
                    cv2.rectangle(Forvideo,(bbox[1],bbox[0]),(bbox[3],bbox[2]),color,3)
                    
                    outBinary_tracking.write(Forvideo)
                    

                    # img_reconstructed
                    frame_num = track_final[idx].tot_frame[i-1] ## Temporal fix

                    frame_name = "Obj_frame"+str(frame_num + NumberFirstImage)+".jpg"
                    Binary_frame_name = "Binary_Obj_frame"+str(frame_num+ NumberFirstImage)+".png"
                    Background_name = "BG_frame"+str(frame_num+ NumberFirstImage)+".jpg"

                    frame_path = os.path.join(track_folder_Original, frame_name)

                        
                    
                    cv2.imwrite(frame_path, img)
                    
                    img_binary =  np.where(img_binary != 0, 255, 0)
                    cv2.imwrite(os.path.join(track_folder_Foreground, Binary_frame_name), (img_binary).astype(np.uint8))

                    cv2.imwrite(os.path.join(track_folder_Background, Background_name), img_Background)
            outBinary_tracking.release()
                    
            
            ## ANOTHER for the bbox imaGes, as its possible for them to hAve different lenGths
            for i in range(len(track_final[idx].image)):
                
                if i == 0 : ## Temporal fix, THE FIRST FRAME IN ALL TRACKS IS ADDED TWO TIMES AT THE BEGINNING, NEED TO FIND WHERE
                    continue
                else:
                    Detection_Reconstructed = track_final[idx].image[i]
                    frame_num = track_final[idx].tot_frame[i-1] ## Temporal fix
                    frame_name = "Detection_Reconst_frame"+str(frame_num + NumberFirstImage)+".png"
                    

                    bbox = track_final[idx].bbox[i]
                    y_min, x_min, y_max, x_max = bbox # Its inversed
                    NewRow = {
                    headers[0]: int(frame_num + NumberFirstImage),
                    headers[1]: (x_min, y_min, x_max, y_max)
                    }

                    # Check if the centroid index is within bounds and add it if it is
                    if i < len(track_final[idx].centroids): 

                        centroid = (int(track_final[idx].centroids[i][0]),int(track_final[idx].centroids[i][1]))
                        NewRow[headers[2]] = centroid
                        bbox = track_final[idx].bbox[i]
                        

                    bbox_df = bbox_df._append(NewRow, ignore_index=True)

                    cv2.imwrite(os.path.join(track_final_images_folder,"Foreground",f"track{idx}",frame_name), Detection_Reconstructed)

        bbox_df.to_excel(os.path.join(track_final_images_folder, "Bbox_labels.xlsx"), index=False)


        # For fish
        for idx in range(len(fish)):

            fish_folder_Original = os.path.join(fish_images_folder,"Original", f"fish{idx}")
            fish_folder_Foreground = os.path.join(fish_images_folder,"Foreground", f"fish{idx}")
            fish_folder_Background = os.path.join(fish_images_folder,"Background", f"fish{idx}")

            os.makedirs(fish_folder_Original, exist_ok=True)
            os.makedirs(fish_folder_Foreground, exist_ok=True)
            os.makedirs(fish_folder_Background, exist_ok=True)
            
            for i in range(len(fish[idx].img_original)):

                img = fish[idx].img_original[i]
                img_binary = fish[idx].img_Complete_binary[i]
                img_Background = fish[idx].img_Background_Estimation[i]

                
                frame_num = fish[idx].tot_frame[i]

                frame_name = "Original_Fish_frame"+str(frame_num-1 + NumberFirstImage)+".jpg"  ## Temporal fix -> frame_num-1
                Binary_frame_name = "Binary_Fish_frame"+str(frame_num-1+ NumberFirstImage)+".png"
                Background_name = "BG_Fish_frame"+str(frame_num-1+ NumberFirstImage)+".jpg"

                frame_path = os.path.join(fish_folder_Original, frame_name)
                    
                cv2.imwrite(frame_path, img)   


                cv2.imwrite(os.path.join(fish_folder_Foreground, Binary_frame_name), img_binary)

                cv2.imwrite(os.path.join(fish_folder_Background, Background_name), img_Background)

            for i in range(len(fish[idx].image)):
                
                Detection_Reconstructed = fish[idx].image[i]

                frame_num = fish[idx].tot_frame[i]

                frame_name = "Fish_Detection_Reconst_frame"+str(frame_num-1+ NumberFirstImage)+".png"
                    
                

                cv2.imwrite(os.path.join(fish_images_folder,"Foreground",f"fish{idx}",frame_name), Detection_Reconstructed)




    if affichage :  ## For the deformation images and the fish gif
        for ff in range(0,len(fish)):
            # plt.figure()
            # plt.imshow(fish[ff].deformation)
            
            outFish = []
            # imgdeformation = fish[ff].deformation
            # plt.figure()
            # plt.imshow(imgdeformation)
            # plt.savefig(os.path.join(full_path,'deformation_'+str(ff)+'.png'),bbox_inches='tight',dpi=100)
            # plt.close('all')
            # imgdeformation = cv2.imread(os.path.join('Test','deformation_'+str(ff)+'.png'))
            # resize_ratio = int(imgdeformation.shape[0]/(500/imgdeformation.shape[1]))
            
            # imgdeformation = cv2.resize(imgdeformation, (500,resize_ratio), interpolation= cv2.INTER_LINEAR)
            
            for tr in range(len(fish[ff].tot_frame)):
                
                imgExplanation_to_write = np.zeros((1200,920,3)) ## Creating the empty image for the fish .gif
                
                ## Adding the window original window of the detected fish candidate
                img = fish[ff].image[tr]
                if img.shape[0]<300:
                    img = np.pad(img,((300-img.shape[0],0),(0,0)), constant_values=((0,0),(0,0)))
                if img.shape[1]<300:
                    img = np.pad(img,((0,300-img.shape[1]),(0,0)), constant_values=((0,0),(0,0)))
                if img.shape[1]>200:
                    a = int((img.shape[1]-200)/2)
                    img = img[:,a:200+a]
                if img.shape[0]>200:
                    a = int((img.shape[0]-200)/2)
                    img = img[a:a+200,:]
                img = img.astype(np.uint8)
                img = cv2.resize(img, (400,400), interpolation= cv2.INTER_LINEAR)
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
                plt.figure()
                plt.imshow(img)
                imgExplanation_to_write[800:1200,520:920,:]=img
                
                ## Adding the fish window to reconstruct
                img = fish[ff].img_to_reconstruct_binary[tr]
                if img.shape[0]<300:
                    img = np.pad(img,((300-img.shape[0],0),(0,0)), constant_values=((0,0),(0,0)))
                if img.shape[1]<300:
                    img = np.pad(img,((0,300-img.shape[1]),(0,0)), constant_values=((0,0),(0,0)))
                if img.shape[1]>200:
                    a = int((img.shape[1]-200)/2)
                    img = img[:,a:200+a]
                if img.shape[0]>200:
                    a = int((img.shape[0]-200)/2)
                    img = img[a:a+200,:]
                img = img.astype(np.uint8)
                img = cv2.resize(img, (400,400), interpolation= cv2.INTER_LINEAR)
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
                imgExplanation_to_write[400:800,520:920,:]=img
                
                ## Adding the reconstructed fish window
                img = fish[ff].img_to_reconstruct[tr]
                if img.shape[0]<300:
                    img = np.pad(img,((300-img.shape[0],0),(0,0)), constant_values=((0,0),(0,0)))
                if img.shape[1]<300:
                    img = np.pad(img,((0,300-img.shape[1]),(0,0)), constant_values=((0,0),(0,0)))
                if img.shape[1]>200:
                    a = int((img.shape[1]-200)/2)
                    img = img[:,a:200+a]
                if img.shape[0]>200:
                    a = int((img.shape[0]-200)/2)
                    img = img[a:a+200,:]
                img = img.astype(np.uint8)
                img = cv2.resize(img, (400,400), interpolation= cv2.INTER_LINEAR)
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            
                imgExplanation_to_write[0:400,520:920,:]=img
                
                
                
                #imgExplanation_to_write[300:300+imgdeformation.shape[0],0:500,:]=imgdeformation
                plt.figure()
                plt.imshow(imgExplanation_to_write)
                outFish.append(cv2.cvtColor(imgExplanation_to_write.astype(np.uint8), cv2.COLOR_BGR2RGB))
            imageio.mimsave(os.path.join(full_path,'Fish'+str(ff)+'_test.gif'), outFish, loop=4, duration = 0.6)      
    
    t2 = datetime.datetime.now()
    print(t2-t1)

if __name__ == "__main__":

    df = pd.read_excel(r'C:\Users\chapi\Documents\STAGE\CODE\bgslibrary\ACtoolbox-main\Organized_Labels.xlsx')
 #to process faster, only the ones that have at least a eel according to manual annoptations from operators
    #filtered_df0 = df[df['Eel_TimeStamp'] != '[]']
    #filtered_df1 = filtered_df0[df['Eel_TimeStamp'] != '['']']

    filtered_df0 = df[df['OtherFish_TimeStamp'] != '[]']
    filtered_df1 = filtered_df0[df['OtherFish_TimeStamp'] != '['']']

    Videoname = filtered_df1['VideoName'].tolist()
    video_paths = []
    for video in Videoname:
        date = video.split("_")[0]
        date = date.replace("-", "_")

        video_paths.append(os.path.join(r'E:\Data',str('ARIS_'+ date+'_AVI'), str(video+'.avi')))

    SeluneVideos = glob.glob(r"ACtoolbox-main\Dataset\Small_ARIS_SELUNE\DonneesARIS_Azenor\SIL\*.avi")
    Missingvideos = glob.glob(r"C:\Users\chapi\Documents\STAGE\CODE\bgslibrary\To extract bgs\*")
    video_paths = Missingvideos
    print(video_paths)
    VideoBool = True
    dataset = "ARIS_Mauzac_OtherFish"
    """"
    Available bgs
    +FrameDifference() +StaticFrameDifference() +WeightedMovingMean() +WeightedMovingVariance()
    +AdaptiveBackgroundLearning() +AdaptiveSelectiveBackgroundLearning() +MixtureOfGaussianV2()
    +PixelBasedAdaptiveSegmenter() +SigmaDelta() +SuBSENSE() +LOBSTER() +PAWCS() +TwoPoints() +ViBe()
    +CodeBook() +FuzzySugenoIntegral() +FuzzyChoquetIntegral() +LBSimpleGaussian() +LBFuzzyGaussian()
    +LBMixtureOfGaussians() +LBAdaptiveSOM() +LBFuzzyAdaptiveSOM() +TBackground() +TBackgroundVuMeter()
    +VuMeter() +KDE() +IndependentMultimodal() +KNN()
    """
    # cv2.createBackgroundSubtractorMOG2(varThreshold=config.seuil_mean,history=int(14*fps),detectShadows = False)
    
    Filter ="Gaussian" #Mean Gaussian Median
    #str(BGS_Method).split(".")[-1].split(" ")[0]
    BGS_Method = bgs.SuBSENSE()
    #2014-11-05_184000 2014-11-16_002000
    #ImagesList = glob.glob(r"C:\Users\chapi\Documents\STAGE\CODE\bgslibrary\ACtoolbox-main\Dataset\Small_ARIS_Mauzac\TEST\All_Originals\2014-11-16_002000_t6_Obj_frame*.jpg")
    # 2014-11-05_184000_t0_Obj_frame*.jpg
    
    #print(len(ImagesList))
    #xx= ImagesList[0].split("\\")[-1].split("_")
    #print(ImagesList[0].split("\\")[-1].split("_"))
    #print(str(xx[0]+"_"+xx[1]+"_"+xx[2]))
    
    #video_paths = SeluneVideos
    if VideoBool:
        for i in range(len(video_paths[:])) :
            BGS_Method = bgs.SuBSENSE()
            main(video_paths[:][i], dataset, VideoBool, BGS_Method, [0], Filter)
            print(f"Videos completed : {i+1} out of {len(video_paths)}")
    else:

        main(video_paths, dataset, VideoBool, BGS_Method, ImagesList, Filter)
    
    
