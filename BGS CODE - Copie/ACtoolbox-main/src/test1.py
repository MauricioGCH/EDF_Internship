
import numpy as np
from scipy.spatial import distance_matrix
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from skimage.morphology import binary_closing, skeletonize
from skimage import io







image = cv2.imread(r"C:\Users\chapi\Documents\STAGE\CODE\bgslibrary\ACtoolbox-main\Dataset\Small_ARIS_Mauzac\TEST\All_Originals\NewMasks\2014-11-06_043000_t0_Obj_frame4098\m_Eel0_2014-11-06_043000_t0_Obj_frame4098.jpg")

print(np.unique(image))
non_zero_sum = np.sum(image[image != 0])
print(non_zero_sum)
plt.imshow(image)
plt.show()
#binary_image =  np.where(image != 255, 0, 255)
#binary_image = ndimage.binary_fill_holes(binary_image).astype(int)
#binary_image = binary_closing(binary_image, footprint=np.ones((3,3)))

#binary_image =  np.where(binary_image == 1, 255, 0)

print(np.unique(binary_image))
plt.imshow(binary_image)
plt.show()

output_path = 'output_image.png'  # Path where you want to save the image
io.imsave(output_path, binary_image)

reviewimage = cv2.imread(output_path)

print(np.unique(reviewimage))
plt.imshow(reviewimage)
plt.show()
