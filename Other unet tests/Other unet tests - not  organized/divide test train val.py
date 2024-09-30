# divide test train val

import numpy
import os
import glob

# counted from the masks as several objects can be tracked
def classCountPerVideo(Videopath, dict, keys):
    Masks_list = glob.glob(os.path.join(Videopath,"Foreground","t*","m_*")) # it doesnt matter to differentiate tracks as tracks from video cant be in different aprts of the dataset
    #initialiwe dictionary for count
    

    for key in class_key:

        # Initialize counter
        count = 0

        # Iterate over the list and count occurrences of the substring
        for filename in Masks_list:
            if key in filename:
                count += 1
        class_dicc[key] = count
    #print(class_dicc)
    return class_dicc

class_key = ['Eel', 'Eel_Arcing', 'SmallFish', 'SmallFish_Arcing', 'Trash', 'Trash_Arcing']
class_dicc = {key: 0 for key in class_key}
#classCountPerVideo(r"C:\Users\d42684\Documents\STAGE\CODES\Small_ARIS_Mauzac\2014-11-16_002000",class_dicc, class_key)



ForallVideosTrain = r"Small_ARIS_Mauzac_UnetReady_Final\Train\2014*"
ForallVideosVal = r"Small_ARIS_Mauzac_UnetReady_Final\Val\2014*"
ForallVideosTest = r"Small_ARIS_Mauzac_UnetReady_Final\Test\2014*"

def Train_Val_Test(All_Video_paths, keys):

    All_videos_path = glob.glob(os.path.join(All_Video_paths))# list of video paths, just until the video
    class_dicc = {key: 0 for key in class_key}
    for video in All_videos_path:
        class_diccInt= classCountPerVideo(video,class_dicc, keys)

        for key in class_dicc:
            class_dicc[key] = class_diccInt[key] + class_dicc[key]
    
    print(class_dicc)

    # for key in keys:

    #     class_threshold = int(class_dicc[key]*0.8)


Train_Val_Test(ForallVideosTrain,class_key)
Train_Val_Test(ForallVideosVal,class_key)
Train_Val_Test(ForallVideosTest,class_key)
