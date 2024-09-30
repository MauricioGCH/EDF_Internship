#verify data
import os
import glob


masklist = glob.glob(r"C:\Users\d42684\Documents\STAGE\CODES\Small_ARIS_Mauzac\TEST\NewMasks\*")
masksBasename = [os.path.basename(path) for path in masklist]

imagelist = glob.glob(r"C:\Users\d42684\Documents\STAGE\CODES\Small_ARIS_Mauzac\TEST\All_Originals\*")
imageBasename = [os.path.basename(path) for path in imagelist]
status = "good"

for maskbasename in masksBasename:
    found = False

    for imgname in imageBasename:
        imgname = imgname[:-4] #.jpg 
        if maskbasename == imgname:
            found = True
    
    if not found :
        print("there is mask that has no image")
        status = "bad"
print(status)

    