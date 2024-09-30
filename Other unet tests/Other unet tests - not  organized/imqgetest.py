import cv2
import numpy as np  
import os
path = r"Small_ARIS_Mauzac_UnetReady\Train\2014-11-15_234000\Foreground\t8\crop_m_Trash0_2014-11-15_234000_t8_Obj_frame1460.jpg"
normalized_path = os.path.join(path)
s = os.path.exists(normalized_path)
print("El path : ",s)

imqge = cv2.imread(normalized_path)
#visualization_epoch_0_batch_0.png
#img2 = cv2.imread("visualization_epoch_0_batch_0.png")

print(imqge)