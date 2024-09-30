##
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2




sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")



mask_generator = SamAutomaticMaskGenerator(sam)

image = cv2.imread('Fish_Frame_1433.png')
masks = mask_generator.generate(image)



print(len(masks))
print(masks[0].keys())


plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show() 