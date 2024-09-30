# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:56:20 2024

@author: D74179
"""

from skimage.morphology import disk
import cv2
import numpy as np
from skimage.measure import label,regionprops
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

class Candidate():
    def __init__(self,track, numero_img):
        # Initlization 
        self.centroids = track.centroids[numero_img[0]:numero_img[-1]]
        self.length = track.length[numero_img[0]:numero_img[-1]]
        self.tot_frame = track.tot_frame[numero_img[0]:numero_img[-1]]
        self.image = track.image[numero_img[0]:numero_img[-1]]
        self.img_shape = track.image[numero_img[0]].shape # Should save the shape of each of the windozs for her thing to work
        indice_nb_frame_o = np.argmin([np.abs(self.length[ii]-np.quantile(self.length,0.75)) for ii in range(len(self.length))]) ##
        self.nb_frame_o = track.tot_frame[numero_img[0]]+indice_nb_frame_o
        self.indice_img = indice_nb_frame_o
        region = regionprops(label(self.image[indice_nb_frame_o]))
        self.major_axis_length = region[0].major_axis_length
        self.minor_axis_length = region[0].minor_axis_length
        self.FPS_rate=track.FPS_rate
        self.length = []
        self.area = []
        self.eccentricities = []
        self.deformation = []
        self.orientation = []

        self.img_original = []##
        self.img_original = track.img_original[numero_img[0]:numero_img[-1]]##

        self.img_Complete_binary = []##
        self.img_Complete_binary = track.img_Complete_binary[numero_img[0]:numero_img[-1]]##

        self.img_Background_Estimation = []##
        self.img_Background_Estimation = track.img_Background_Estimation[numero_img[0]:numero_img[-1]]##
        
        
        
        self.image = track.image[numero_img[0]:numero_img[-1]]
        self.img_to_reconstruct_binary = track.img_to_reconstruct_binary[numero_img[0]:numero_img[-1]]
        self.img_to_reconstruct = track.img_to_reconstruct[numero_img[0]:numero_img[-1]]
        
        
        
        
        if track.centroids[numero_img[0]][0]-track.centroids[numero_img[-1]][0]>0:
            self.sens = 'right-to-left'
        else:
            self.sens = 'left-to-right'
            
    def add_detection(self,length,area,eccentricities,orientation):
        self.length.append(length)
        self.area.append(area)
        self.eccentricities.append(eccentricities)
        self.orientation.append(orientation*180/np.pi)
    
    def redress_image(self):
        valeur_orientation = np.mean(np.abs(self.orientation))
        #if valeur_orientation<80 :
        sign = np.mean(self.orientation)/np.abs(np.mean(self.orientation))
        for ii in range(len(self.image)):
            rot_mat = cv2.getRotationMatrix2D(center = (int(self.image[ii].shape[0]/2),int(self.image[ii].shape[1]/2)), 
                                              angle = sign*(90-valeur_orientation), scale = 1)
            self.image[ii] = cv2.warpAffine(self.image[ii],M=rot_mat,dsize=self.image[ii].shape) # For gray scale image
                
        
        
    def add_deformation(self,reconstruction):
        self.deformation = reconstruction
   



def detectROI(binaryMask,length_px,long_px): ## Write about this in the logbook
    "It crops the regions of interest in an image, using regionpropos and label. It filters not interesting regions "
    "by conditions of min and max height and width. It is enough if it passes one of them as the object orientation may change"
    k1 = int(85/long_px)+1
    maxAreaMask = []
    ##TODO Add intermediate here of the dilation to see if its ideal
    regionBinaryMask = regionprops(label(cv2.dilate(binaryMask,np.ones((k1,k1)),iterations = 1)),cache=False) # dilate for broke objects, label and regionsprops
    #regionBinaryMask = regionprops(label(binaryMask))


    for ii in range(len(regionBinaryMask)):
        #print(np.unique(binaryMask))
        tovisualize = cv2.cvtColor(binaryMask, cv2.COLOR_GRAY2BGR)

        centroid = (int(regionBinaryMask[ii].centroid[1]),int(regionBinaryMask[ii].centroid[0]))

        cv2.circle(tovisualize,centroid, 3,(0,0,255),-1)

        #xx = regionBinaryMask[ii].image.astype(np.uint8) * 255
        #plt.imshow(tovisualize) ##  YA REVISE QUE EN UN SOLO FRAME ESTA VAINA SI DETECTA CORRECTAMENTE LAS DIFERENTES REGIONES;eNTONCES TOCA MIRAR LUEGO QUE PASA CON LAS OTRAS REGIONES.
        #plt.show()
        height_region = regionBinaryMask[ii].bbox[2]-regionBinaryMask[ii].bbox[0]
        width_region = regionBinaryMask[ii].bbox[3]-regionBinaryMask[ii].bbox[1]
        condition_height = height_region>=0.6*length_px and height_region<=2*length_px
        condition_width = width_region>=0.6*length_px and width_region<=2*length_px
        if condition_height or condition_width: ## it doesn't consider if the eel is diagonal
            print(height_region, 0.6*length_px, 2*length_px)
            print("-----")
            print(width_region, 0.6*length_px, 2*length_px)
            maxAreaMask.append(ii)
    
    if maxAreaMask:
        regionsCandidate = [regionBinaryMask[i] for i in maxAreaMask]
        return True,regionsCandidate
    else:
        return False,None
    
def select_candidate_img(track):
    quantile_length = np.quantile(track.length,0.75) ## What is this about the quantile
    subsection = []
    subsection_tot =[]
    res = False
    candidate = []
    for ii in range(len(track.length)):
        if track.length[ii]/quantile_length>=0.7 and track.length[ii]/quantile_length<=1.3:
            subsection.append(ii)
        else:   
            if len(subsection)!=0:
                subsection_tot.append(subsection)
                subsection = []
    if len(subsection_tot)!=0:
        indice_suite = np.argmax([len(jj) for jj in subsection_tot])
        if len(subsection_tot[indice_suite])>3:
            candidate = Candidate(track,subsection_tot[indice_suite])
            res = True
    return candidate,res

def img_reconstruction(vignette_o,orientation):
    epsi_dilate=11
    espi_dilate2 = 5
    e1 = 1
    e2 = 3

    # Defining the structuring elements for all of the possible cases    
    if 90-np.abs(orientation*180/np.pi)<45: # Case horizontal
        element = np.ones((espi_dilate2,epsi_dilate))
        noise = np.ones((e1,e2))
    else:
        element = np.ones((epsi_dilate,espi_dilate2))
        noise = np.ones((e1,e2))
    
    if len(vignette_o)!=0 and len(vignette_o[0])!=0:
       
        
        closing = cv2.morphologyEx(vignette_o, cv2.MORPH_CLOSE, noise)#np.ones(espi_morpho))#disk(2)) ## is this to delete the noise ? You're using a elemnt of 1x3 called noise
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, noise)#np.ones(espi_morpho)) ## is this to delete the noise ? You're using a elemnt of 1x3 called noise
        vignette = opening
        
        
        if np.abs(orientation*180/np.pi)>=0 and np.abs(orientation*180/np.pi)<=30:#90//np.abs(orientation*180/np.pi)==0: # Verticale
            # print('vertical')
            
            element = np.ones((epsi_dilate,espi_dilate2))
        if  np.abs(orientation*180/np.pi)>30 and np.abs(orientation*180/np.pi)<=60:#90//np.abs(orientation*180/np.pi)==1: # Cas de biais
            # print('biais')
            element = np.ones((epsi_dilate,epsi_dilate)) 
            
        if np.abs(orientation*180/np.pi)>60 and np.abs(orientation*180/np.pi)<=90:#90//np.abs(orientation*180/np.pi)==2: # Case horizontal
            # print('horizontal')
            element = np.ones((espi_dilate2,epsi_dilate))
      
            
        vignette_f = cv2.dilate(vignette,element)
        region = regionprops(label(vignette_f))

        if region:
            ind_max = np.max([rr.area for rr in region])
        else:
            ind_max = 0  # Or any default value you want to assign for an empty array
        #ind_max = np.max([rr.area for rr in region])
        for reg in region :
            if reg.area <ind_max:
                for coords in reg.coords:
                    vignette_f[coords[0]][coords[1]]=0
                         
        vignette_f = vignette_f/255+vignette_o/255
        
        _,vignette_f = cv2.threshold(vignette_f,1,1,cv2.THRESH_BINARY) # To only keep pixels that were present in both vignettes. An intersection, basically.
                    

        vignette_final = cv2.dilate(vignette_f,disk(epsi_dilate))
        vignette_final = cv2.erode(vignette_final,disk(epsi_dilate-1))
        vignette_final = cv2.dilate(vignette_final,np.ones((3,3)))
        
        skeleton = skeletonize(vignette_final)
        length = np.sum(skeleton)

        return 255*vignette_final, length