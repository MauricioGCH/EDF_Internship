# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:54:17 2024

@author: D74179
"""

import numpy as np
from skimage.measure import label,regionprops
from skimage.morphology import skeletonize_3d
from sklearn.cluster import KMeans


def calculation_candidate_morphology(candidate):
    
    for ii in range(len(candidate.image)):
        # Length calculation
        img_ske = skeletonize_3d(candidate.image[ii]/255) ## deprecareted change to skeletonize
        img_s = np.zeros(candidate.image[ii].shape)
        for ll in range(len(img_s)):
            for jj in range(len(img_s[0])):
                if img_ske[ll][jj] == True:
                    img_s[ll][jj] = 1
        length = np.sum(img_s)
        
        # Area calculation ## Here it is assumed that all reagions were succesfully connected
        reg = regionprops(label(candidate.image[ii]),cache=False)
        area = reg[0].area 
        orientation = reg[0].orientation
        
        # Eccentricities calculation
        coord = []
        
        for cc in reg[0].coords:
            coord.append([cc[0],cc[1]])
        kmeans = KMeans(n_clusters=3, random_state=0,init='random').fit(coord) #3
        labels = kmeans.labels_
        togray = np.zeros(candidate.image[ii].shape,dtype='uint8') ## cqhnged from candidate.image_shape, as it only saves the shape of the first candidate
        for cc in range(len(coord)):
            togray[coord[cc][0]][coord[cc][1]] = 50*(labels[cc]+1)
        reg_eccen = regionprops(togray,cache=False)
        eccentricities = [rr.eccentricity for rr in reg_eccen]
        
        candidate.add_detection(length,area,eccentricities,orientation)
    
    
    return candidate