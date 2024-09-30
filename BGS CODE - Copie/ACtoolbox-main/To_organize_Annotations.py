
import os
import glob
import shutil
#absolute_path = os.path.dirname(__file__)

path = "C:\\Users\\d42684\\Documents\\STAGE\\CODES\\ACtoolbox-main\\Dataset\\Small_ARIS_Mauzac\\TEST"

AllVideoPaths = glob.glob(os.path.join(path,"*"))

for Video in AllVideoPaths:

    VideoName = Video.split("\\")[-1]

    Originalfolder = glob.glob(os.path.join(Video,"Track_final_Images","Original","*"))

    for track in Originalfolder:


        TrackNumber =track.split("\\")[-1][-1]

        Images = glob.glob(os.path.join(track,"*"))

        for Image in Images:

            ImageName = Image.split("\\")[-1]

            destination = os.path.join(path,"All_Originals",str(VideoName+"_"+"t"+TrackNumber+"_"+ImageName))

            shutil.move(Image, destination)