#EelDataset

import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import random

def random_rotation(image, mask, degrees):
    angle = random.uniform(-degrees, degrees)
    image = image.rotate(angle, resample=Image.BILINEAR)
    mask = mask.rotate(angle, resample=Image.NEAREST)
    return image, mask

class CustomDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, degrees = 45):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
        self.degrees = degrees

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Read mask (target)
        mask = cv2.imread(self.mask_paths[idx]) # list index out of range

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        _,mask = cv2.threshold(mask,127,1, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)
        
        # Read image (input)
        image = cv2.imread(self.image_paths[idx])
        #print(image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)

        # Resize both image and mask to 150x150
        image=cv2.resize(image, (150, 150))
        mask=cv2.resize(mask, (150, 150))

        #TO Image PIL for rotation
        image = Image.fromarray(image , mode="RGB")
        mask = Image.fromarray(mask)
        image, mask = random_rotation(image, mask, self.degrees)

        VisualizationOnly = np.array(image)
        

        
        

        mask = transforms.functional.pil_to_tensor(mask)
        mask = mask.float()

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
            #print(image.size())
        

        return VisualizationOnly, image, mask
