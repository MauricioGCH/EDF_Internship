

import os
import glob
import pandas as pd
import shutil

base_path = r'NewMasks4'


ListOfmasksframes = glob.glob(r'NewMasks4\2014*')

for i in range(len(ListOfmasksframes)):
    
    ActualBasenameSplit = os.path.basename(ListOfmasksframes[i]).split("_")
    ActualVideo = ActualBasenameSplit[0] + "_" + ActualBasenameSplit[1]
    ActualtrackOfVideo = ActualBasenameSplit[2]


    paths = glob.glob(base_path+"\\"+"*"+os.path.basename(ListOfmasksframes[i]))
    # FOR MASKS
    try:
        assert os.path.basename(paths[0]) in os.path.basename(paths[1])

        os.makedirs(os.path.join(base_path,ActualVideo,"Foreground",ActualtrackOfVideo), exist_ok= True)
        os.makedirs(os.path.join(base_path,ActualVideo,"Original",ActualtrackOfVideo), exist_ok= True)

        shutil.move(paths[1], os.path.join(base_path,ActualVideo,"Foreground",ActualtrackOfVideo,os.path.basename(paths[1])))
        shutil.move(paths[0], os.path.join(base_path,ActualVideo,"Original",ActualtrackOfVideo,os.path.basename(paths[0])))
    except:
        print(" missing mask or dobled original : ", paths)




    
    #os.makedirs(os.path.join(base_path,ActualVideo,"Original",ActualtrackOfVideo), exist_ok= True)
    #shutil.copy(os.path.join(base_path,"All_Originals",str(os.path.basename(ListOfmasksframes[i]))), os.path.join(base_path,ActualVideo,"Original",ActualtrackOfVideo,str(os.path.basename(ListOfmasksframes[i])+".jpg")))