import numpy as np

import os



rangeCameraMin = 0.2
rangeCameraMax = 10
long = 600
seuil_mean = 30

absolute_path = os.path.dirname(__file__)
relative_path = "Dataset"
full_path = os.path.join(absolute_path, relative_path)

Specific_Dataset = 'ARIS_Mauzac' #ARIS_Mauzac# BV_Mauzac

pathAVI = os.path.join(full_path,Specific_Dataset)#'C:/Users/D74179/Documents/BIODIV/Mauzac_ARIS/'
videoName = '2014-11-15_210000' # 2019-05-06_032000

# Configuration of the Kalman filter
deltaT=1/7
std_acc=10

Q = np.array([[(deltaT**4)/4,0,(deltaT**3)/2,0],[0,(deltaT**4)/4,0,(deltaT**3)/2],[(deltaT**3)/2,0,deltaT**2,0],[0,(deltaT**3)/2,0,deltaT**2]])*(std_acc**2)
B = np.array([[0.5*(deltaT**2),0],[0,0.5*(deltaT**2)],[deltaT,0],[0,deltaT]])
R = np.diag([0.1,0.5])
A = np.array([[1,0,deltaT,0],[0,1,0,deltaT],[0,0,1,0],[0,0,0,1]])
H = np.array([[1,0,0,0],[0,1,0,0]])
epsi_frame = 7
epsi_orientation = 25

