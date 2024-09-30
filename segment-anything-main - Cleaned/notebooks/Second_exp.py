## Experiment prompts from VGG to sam
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import json
from skimage.measure import label, regionprops
import glob
import os
from PIL import Image
import torch
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to sys.path
sys.path.append(parent_dir)

# Now you can import the module from the parent directory

from segment_anything.utils.transforms import ResizeLongestSide


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    




def prepare_image(image, transform):
    image = transform.apply_image(image)
    image = torch.as_tensor(image) 
    return image.permute(2, 0, 1).contiguous()


import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "C:\\Users\\d42684\\Documents\\STAGE\\CODES\\segment-anything-main\\sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
#sam.to(device=device)

predictor = SamPredictor(sam)




ImageList = glob.glob(os.path.join('C:\\Users\\d42684\\Documents\\STAGE\\CODES\\ACtoolbox-main\\Dataset\\Small_ARIS_Mauzac\\TEST\\All_Originals\\*.jpg'))

df = pd.read_csv(r'C:\Users\d42684\Documents\STAGE\CODES\segment-anything-main\notebooks\SecondVersion_Annotations_Filled.csv')

filtered_df0 = df[df['file_attributes'] != '{"Object_Count":"0"}']

for imagepath in ImageList[397:]: # Puedo optimizar al directamente eliminar todos los que tienen region 0

    imageName = imagepath.split('\\')[-1]


    filtered_df = filtered_df0[filtered_df0['filename'] == imageName]

    # Check if filtered_df is empty
    if filtered_df.empty:
        continue  # Skip this iteration if no matches found



    print("File attributes (regions):")
    print(json.loads(filtered_df.iloc[0]['file_attributes'])['Object_Count'])



    image = cv2.imread(imagepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #predictor.set_image(image)


    
    print("image atributes: ")
    print(filtered_df["region_shape_attributes"])
    print('-----------------------------------------')

    cx = []
    cy = []

    x0 = []
    y0 = []
    y1 = []
    x1 = []

    for i in range(len(filtered_df)):

        info = json.loads(filtered_df.iloc[i]['region_shape_attributes'])

        if info['name'] == 'point':

            cx.append(info['cx'])
            cy.append(info['cy'])
            #print(f"Point - cx: {cx}, cy: {cy}")

        elif info['name'] == 'rect':
            # Process rectangle information
            x0.append(info['x'])
            y0.append(info['y'])
            x1.append(int(info['x']) + int(info['width']))
            y1.append(int(info['y']) + int(info['height']))
            #print(f"Rectangle - x: {x0}, y: {y0}, x1: {x1}, y1: {y1}")
            #print('wait')
    
    input_label = np.ones(len(cx))
    Points_array = torch.tensor(np.column_stack((np.array(cx), np.array(cy))))
    Bbox_array = torch.tensor(np.column_stack((x0, y0, x1, y1)))



    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)


    # batched_input = [
    #  {
    #      'image': prepare_image(image, resize_transform),
    #      'boxes': resize_transform.apply_boxes_torch(Bbox_array, image.shape[:2]),
    #      'point_coords': resize_transform.apply_coords_torch(Points_array, image.shape[:2]),
    #      'point_labels' : input_label,
    #      'original_size': image.shape[:2]
    #  }
    # ]

    batched_input = [
     {
         'image': prepare_image(image, resize_transform),
         'boxes': resize_transform.apply_boxes_torch(Bbox_array, image.shape[:2]),
         
         'original_size': image.shape[:2]
     }
    ]
    
    batched_output = sam(batched_input, multimask_output=False)

    
    
    

    plt.imshow(image)
    for mask in batched_output[0]['masks']:
        show_mask(mask.numpy(), plt.gca(), random_color=True)
    # for box in Bbox_array:
    #     show_box(box.numpy(), plt.gca())
    plt.axis('off')


    plt.tight_layout()
    plt.show()
    # masks, scores, logits = predictor.predict(
    #     point_coords=input_point,
    #     point_labels=input_label,
    #     multimask_output=True,   
    # )
    print("wait")
    