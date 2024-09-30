# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:55:53 2024

@author: D74179
"""

import cv2
import numpy as np
from skimage.measure import label,regionprops

def func_ratio_px(img,range_min,range_max): # Calculation of the length and area represented by a pixel in mm and mm²
    _,thresh = cv2.threshold(img,2,1,cv2.THRESH_BINARY)
    reg = regionprops(label(thresh),cache=False)
    ind_reg = np.argmax([len(r.coords) for r in reg])
    y_min,y_max = np.min([r[1] for r in reg[ind_reg].coords]),np.max([r[1] for r in reg[ind_reg].coords])
    y_mil = int(((y_max-y_min)/2) + y_min) #y milleu/middle
    coord_mil = [r[0] for r in reg[ind_reg].coords if r[1]==y_mil] # x coordinates that match the y_mil
    x_min,x_max = np.min(coord_mil), np.max(coord_mil)
    ratio_px = (x_max - x_min)/ (range_max - range_min)
    range_manquant = range_min*ratio_px

    pixel_val = np.round((1/ratio_px)*1000)
    area_px = pixel_val**2
    return (area_px),x_max,int(range_manquant),y_mil,pixel_val