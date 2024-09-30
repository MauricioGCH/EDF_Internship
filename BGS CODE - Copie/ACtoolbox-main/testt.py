
import ctypes
import glob
import matplotlib.pyplot as plt

ctypes.cdll.LoadLibrary("C:\\OpenCV490\\build\\x64\\vc16\\bin\\opencv_world490.dll")
import cv2

import build_win64.pybgs as bgs




# Initialize the background subtraction algorithm
algorithm = bgs.LOBSTER()

video_file = r"C:\Users\chapi\Documents\STAGE\CODE\bgslibrary\ACtoolbox-main\Dataset\Small_ARIS_Mauzac\TEST\All_Originals\2014-11-05_184000_t0_Obj_frame*.jpg"
#2014-11-05_184000_t0_Obj_frame214.jpg
Imagelist = glob.glob(video_file)
i = 0
# Main loop to process the video frames
for framepath in Imagelist:
   print(i)
   i = 1 + i
   frame = cv2.imread(framepath)

   # Apply the background subtraction algorithm

   img_output = algorithm.apply(frame)

   # Retrieve the current background model

   img_bgmodel = algorithm.getBackgroundModel()

   # Display the foreground mask and the background model
   if i > 99:
    fig, axs = plt.subplots(1,3, figsize =(12,6))

    axs[0].imshow(frame, cmap = "gray")
    axs[1].imshow(img_output, cmap = "gray")
    axs[2].imshow(img_bgmodel, cmap = "gray")
    plt.tight_layout()
    plt.show()
    
   