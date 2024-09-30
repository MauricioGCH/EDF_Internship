#Organize individuals
#C:\Users\d42684\Documents\STAGE\CODES\Small_ARIS_Mauzac\2014-11-05_184000\Foreground\t0
import glob
import os
import numpy as np
import cv2
from skimage.measure import label,regionprops
import math
start = 1868
finish = 1896
numbers = list(range(start, finish+1))

Actualcentroid = ()
SameIndividual = []
for i in range(len(numbers)):
    paths = glob.glob(os.path.join(r'C:\Users\d42684\Documents\STAGE\CODES\Small_ARIS_Mauzac\Train\2014-11-17_132000\Foreground\t1',str( "*"+ str(numbers[i]))+"*"  ))

    
    if len(paths) == 0: # the frame doesnt exist
        continue
    
    if i == 0:
        print(i)
        SameIndividual.append(paths[0])
        imagebgr = cv2.imread(paths[0])
        imggray = cv2.cvtColor(imagebgr, cv2.COLOR_BGR2GRAY)
        _,imgbinary = cv2.threshold(imggray,127,255, cv2.THRESH_BINARY)

        Actualprops = regionprops(label(imgbinary))

        if len(Actualprops) == 1 :
            #Easy case
            Actualbbox = Actualprops[0].bbox
            Actualcentroid = Actualprops[0].centroid
            print("entro en el primero")

        elif len(Actualprops) > 1:
            # Biggest bbox
            area = 0
            
            for prop in Actualprops:
                if prop.area_bbox > area:
                    area = prop.area_bbox
                    Actualbbox = prop.bbox
                    Actualcentroid = prop.centroid
                    print("entro en el segundo")
            
            cv2.rectangle(imgbinary, (bbox[1], bbox[0]),(bbox[3],bbox[2]), (255,0,0),3)
            cv2.imshow("image",imgbinary)
            cv2.waitKey(0)
            cv2.destroyAllWindows() 
        
        
    else:
        
        distances = []
        centroids = []
        bboxs = []
        for j in range(len(paths)):
            imagebgr = cv2.imread(paths[j])
            imggray = cv2.cvtColor(imagebgr, cv2.COLOR_BGR2GRAY)
            _,imgbinary = cv2.threshold(imggray,127,255, cv2.THRESH_BINARY)

            Actualprops = regionprops(label(imgbinary))

            if len(Actualprops) == 1 :
                #Easy case
                bbox = Actualprops[0].bbox
                bboxs.append(bbox)
                centroid = Actualprops[0].centroid
                centroids.append(centroid)
            elif len(Actualprops) > 1:
                # Biggest bbox
                area = 0
                bbox = []
                for prop in Actualprops:
                    if prop.area_bbox > area:
                        area = prop.area_bbox
                        bbox = prop.bbox
                        centroid = prop.centroid
                bboxs.append(bbox)
                centroids.append(centroid)
            else:
                print("something wrong, no regions ??")
            
            if len(Actualcentroid) == 0:
                Actualcentroid = centroid
            distances.append(math.dist(Actualcentroid,centroid))
        min_dis = min(distances)

        path_index = distances.index(min_dis)
        Actualcentroid = centroids[path_index]
        Actualbbox = bboxs[path_index]

        SameIndividual.append(paths[path_index])

print(SameIndividual)

import shutil

for path in SameIndividual:
    shutil.move(path, os.path.join(r'C:\Users\d42684\Documents\STAGE\CODES\Small_ARIS_Mauzac\Train\2014-11-17_132000\Foreground\t11',os.path.basename(path) ))

        
    

    

    

