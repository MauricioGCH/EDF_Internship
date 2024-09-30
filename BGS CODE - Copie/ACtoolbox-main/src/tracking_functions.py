# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:28:46 2024

@author: D74179
"""

import numpy as np
import config

class Track():
    def __init__(self,centroid,Q, frame,deltaT,length,img_binary,fps,img_to_reconstruct_binary,img_to_reconstruct,bbox, original, Complete_binary, Background_Estimation):
        # Initlization 
        self.centroids = []
        self.bbox = []
        self.bbox.append(bbox)
        self.Xc = np.array([[[centroid[0]],[centroid[1]],[0],[0]]])
        self.Xp = np.array([[[centroid[0]],[centroid[1]],[0],[0]]])
        self.Pc = np.array([Q])
        self.Pp = np.array([Q])
        self.orientation=[]
        # self.last_frame = frame
        self.tot_frame = []
        self.deltaT = deltaT
        self.FPS_rate = fps
        self.length = []
        self.length.append(length)
        self.sens = 'none'

        self.img_original = []##
        self.img_original.append(original)##

        self.img_Complete_binary = []##
        self.img_Complete_binary.append(Complete_binary)##

        self.img_Background_Estimation = []##
        self.img_Background_Estimation.append(Background_Estimation)##

        self.image = []
        self.image.append(img_binary)
        self.img_to_reconstruct_binary =[]
        self.img_to_reconstruct = []
        self.img_to_reconstruct_binary.append(img_to_reconstruct_binary) ## its binary but cropped
        self.img_to_reconstruct.append(img_to_reconstruct)
        
        
    
    def predict(self,A):
        if len(self.Xp)!=len(self.Xc):
            self.Xc = np.concatenate((self.Xc,[self.Xc[-1]]))
            self.Pc = np.concatenate((self.Pc,[self.Pc[-1]]))
            self.centroids.append([0,0])
        self.Xp = np.concatenate((self.Xp,np.array([np.dot(A,self.Xc[-1])])))
        self.Pp = np.concatenate((self.Pp,np.array([np.dot(np.dot(A,self.Pc[-1]),np.transpose(A))+config.Q])))
        
    def correct_and_add(self,X_obs,A,Q,H,R,frame,orientation,length,img_binary,img_to_reconstruct_binary,img_to_reconstruct,bbox, original, Complete_binary, Background_Estimation):
        self.tot_frame.append(frame)
        self.last_frame = frame
        self.last_position = X_obs
        self.orientation.append(orientation)
        self.length.append(length)
        self.bbox.append(bbox)
        # if  self.img_to_reconstruct[-1].shape ==img_to_reconstruct.shape : # Temporal fix, we should search the reason why the first frame of each track is add 2 times, probably a problem in the
        #     if (self.img_to_reconstruct[-1] !=img_to_reconstruct).all():
        #         self.image.append(img_binary)
        #         self.img_to_reconstruct_binary.append(img_to_reconstruct_binary)
        #         self.img_to_reconstruct.append(img_to_reconstruct)
            
            
        #         self.img_original.append(original)
        self.image.append(img_binary)
        self.img_to_reconstruct_binary.append(img_to_reconstruct_binary)
        self.img_to_reconstruct.append(img_to_reconstruct)
            
            
        self.img_original.append(original) ##
        self.img_Complete_binary.append(Complete_binary) ##
        self.img_Background_Estimation.append(Background_Estimation)##
        
        
        
        # In case the number of observation increase to 2, we can initialize the Kalman filter
        if len(self.tot_frame)==2:
            # Initilization of the Xc and Xp matrix with initial conditions
            vx0 = (X_obs[0]-self.centroids[0][0])/self.deltaT
            vy0 = (X_obs[1]-self.centroids[0][1])/self.deltaT
            self.Xc = np.array([[[self.centroids[0][0]],[self.centroids[0][1]],[vx0],[vy0]]])
            self.Xp = np.array([[[self.centroids[0][0]],[self.centroids[0][1]],[vx0],[vy0]]])
            self.Xc = np.array([[[self.centroids[0][0]],[self.centroids[0][1]],[vx0],[vy0]]])
            self.Xp = np.array([[[self.centroids[0][0]],[self.centroids[0][1]],[vx0],[vy0]]])
            
            
            self.centroids.append(X_obs)
            
            for ii in range(len(self.centroids)):
                self.Xp = np.concatenate((self.Xp,np.array([np.dot(A,self.Xc[-1])])))
                self.Pp = np.concatenate((self.Pp,np.array([np.dot(np.dot(A,self.Pc[-1]),np.transpose(A))+Q])))
                y = np.array([[self.centroids[ii][0]],[self.centroids[ii][1]]])-np.dot(H,self.Xp[-1])
                S = np.dot(np.dot(H,self.Pp[-1]),np.transpose(H))+R
                K = np.dot(np.dot(self.Pp[-1],np.transpose(H)),np.linalg.inv(S))
                self.Xc = np.concatenate((self.Xc,np.array([self.Xp[-1]+np.dot(K,y)])))
                dot1 = np.eye(4)-np.dot(K,H)
                self.Pc = np.concatenate((self.Pc,np.array([np.dot(dot1,self.Pp[-1])])))
            
        else:
            
            y = np.array([[X_obs[0]],[X_obs[1]]])-np.dot(H,self.Xp[-1])
            S = np.dot(np.dot(H,self.Pp[-1]),np.transpose(H))+R
            K = np.dot(np.dot(self.Pp[-1],np.transpose(H)),np.linalg.inv(S))
            self.Xc = np.concatenate((self.Xc,np.array([self.Xp[-1]+np.dot(K,y)])))
            dot1 = np.eye(4)-np.dot(K,H)
            self.Pc = np.concatenate((self.Pc,np.array([np.dot(dot1,self.Pp[-1])])))
           
            self.centroids.append(X_obs)
