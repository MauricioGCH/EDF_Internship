#Masks visualization
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
# Example paths for the images and masks
#2014-11-16_002000_t6_Obj_frame2876  2014-11-16_002000_t8_Obj_frame3062
ImgToSee = '2014-11-16_002000_t6_Obj_frame2876'

image_path = os.path.join(r'C:\Users\d42684\Documents\STAGE\CODES\ACtoolbox-main\Dataset\Small_ARIS_Mauzac\TEST\All_Originals',str(ImgToSee+'.jpg'))
mask1_path = os.path.join(r'C:\Users\d42684\Documents\STAGE\CODES\ACtoolbox-main\Dataset\Small_ARIS_Mauzac\TEST\All_Originals\masks_bbox',str('m_'+ImgToSee+'.jpg'))
mask2_path = os.path.join(r'C:\Users\d42684\Documents\STAGE\CODES\ACtoolbox-main\Dataset\Small_ARIS_Mauzac\TEST\All_Originals\masks_points',str('m_'+ImgToSee+'.jpg'))

# Load images and masks
image = cv2.imread(image_path) #image
mask1 = cv2.imread(mask1_path) # bbox
mask2 = cv2.imread(mask2_path) # points

# Convert masks to 3-channel images for transparency
#mask1_colored = cv2.cvtColor(mask1, cv2.COLOR_GRAY2RGB)
#mask2_colored = cv2.cvtColor(mask2, cv2.COLOR_GRAY2RGB)

# Create figure and subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))


combined_image1 = image.copy()
combined_image2 = image.copy()
alpha = 0.2  # Transparency factor

combined_image1 = cv2.addWeighted(combined_image1, 1, mask1, alpha, 0)
# Superimpose the second mask


# Plot the original image in the first subplot
axs[0].imshow(cv2.cvtColor(combined_image1, cv2.COLOR_BGR2RGB))
axs[0].set_title('Img + mask_bbox')
axs[0].axis('off')

# Plot the image with superimposed masks in the second subplot
combined_image2 = cv2.addWeighted(combined_image2, 1, mask2, alpha, 0)

axs[1].imshow(cv2.cvtColor(combined_image2, cv2.COLOR_BGR2RGB))
axs[1].set_title('Img + mask_points')
axs[1].axis('off')

plt.tight_layout()
plt.show()