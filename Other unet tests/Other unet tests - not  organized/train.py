#training
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from UNet import UNet
import glob
import matplotlib.pyplot as plt
import re
import numpy as np

from EelDataset import CustomDataset

def extract_frame_number(file_path):
    match = re.search(r'_frame(\d+)', file_path)
    if match:
        return int(match.group(1))
    return -1  # Default value if no frame number is found

def visualize_sample(images, masks, outputs, epoch, batch_idx, VisualizationProb, division):
    os.makedirs('Visualize', exist_ok=True)

    images = images
    masks = masks.cpu().numpy()
    outputs = torch.sigmoid(outputs)
     # Adjust this threshold as needed
    VisualizationProb = 0.5
      
    outputs = (outputs >= VisualizationProb).detach().cpu().numpy()

    if division =="train":
        fig, axes = plt.subplots(3, images.shape[0], figsize=(15, 5))
        for i in range(images.shape[0]):
            axes[0, i].imshow(images[i])
            axes[0, i].set_title("Input Image")
            axes[0, i].axis('off')

            axes[1, i].imshow(masks[i, 0], cmap='gray')
            axes[1, i].set_title("True Mask")
            axes[1, i].axis('off')

            axes[2, i].imshow(outputs[i, 0], cmap='gray')
            axes[2, i].set_title("Predicted Mask")
            axes[2, i].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join('Visualize', f'{division}_visualization_epoch_{epoch}_batch_{batch_idx}.png'))
        plt.close()
    elif division =="val":
        fig, axes = plt.subplots(3, figsize=(15, 5))
        
        axes[0].imshow(images[i])
        axes[0].set_title("Input Image")
        axes[0].axis('off')

        axes[1].imshow(masks[i, 0], cmap='gray')
        axes[1].set_title("True Mask")
        axes[1].axis('off')

        axes[2].imshow(outputs[i, 0], cmap='gray')
        axes[2].set_title("Predicted Mask")
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join('Visualize', f'{division}_visualization_epoch_{epoch}_batch_{batch_idx}.png'))
        plt.close()



class RandomRotationWithMask(transforms.RandomRotation):
    def __call__(self, img, mask):
        angle = self.get_params(self.degrees)

        # Apply rotation to image
        img = transforms.functional.rotate(img, angle, self.resample, self.expand, self.center)

        # Apply rotation to mask
        mask = transforms.functional.rotate(mask, angle, self.resample, self.expand, self.center)
    
# Define transformations for image and mask (including resizing)
transform = transforms.Compose([
    #transforms.Resize((150,150)),
    transforms.ToTensor(),  # Convert images to tensors
    
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize images
])


Train = glob.glob(os.path.join(r'Small_ARIS_Mauzac_UnetReady\Train\2014*'))
TrainImages = []
TrainMasks = []

# Iterate over the provided image paths
for VideoPath in Train:

    TrackspathsMasks = glob.glob(os.path.join(VideoPath,"Foreground","t*"))
    TrackspathsOriginals = glob.glob(os.path.join(VideoPath,"Original","t*"))

    if len(TrackspathsMasks) != len(TrackspathsOriginals):
        print("Error : Video doesnt have the same tracks for gt and input")
        break

    for i in range(len(TrackspathsMasks)):

        MasksInTrack = glob.glob(os.path.join(TrackspathsMasks[i],"crop_*"))
        sorted_MasksInTrack = sorted(MasksInTrack, key=extract_frame_number)

        OriginalsInTrack = glob.glob(os.path.join(TrackspathsOriginals[i],"crop_*"))
        sorted_OriginalsInTrack = sorted(OriginalsInTrack, key=extract_frame_number)

        if len(sorted_MasksInTrack) != len(sorted_OriginalsInTrack):
                print("Error : Video doesnt have the same tracks for gt and input")
                break
        
        TrainImages = TrainImages + sorted_OriginalsInTrack
        TrainMasks =  TrainMasks + sorted_MasksInTrack

print(len(TrainImages))
print(len(TrainMasks)) 


Val = glob.glob(os.path.join(r'Small_ARIS_Mauzac_UnetReady\Val\2014*'))
ValImages = []
ValMasks = []

# Iterate over the provided image paths
for VideoPath in Val:

    TrackspathsMasks = glob.glob(os.path.join(VideoPath,"Foreground","t*"))
    TrackspathsOriginals = glob.glob(os.path.join(VideoPath,"Original","t*"))

    if len(TrackspathsMasks) != len(TrackspathsOriginals):
        print("Error : Video doesnt have the same tracks for gt and input")
        break

    for i in range(len(TrackspathsMasks)):

        MasksInTrack = glob.glob(os.path.join(TrackspathsMasks[i],"crop_*"))
        sorted_MasksInTrack = sorted(MasksInTrack, key=extract_frame_number)

        OriginalsInTrack = glob.glob(os.path.join(TrackspathsOriginals[i],"crop_*"))
        sorted_OriginalsInTrack = sorted(OriginalsInTrack, key=extract_frame_number)

        if len(sorted_MasksInTrack) != len(sorted_OriginalsInTrack):
                print("Error : Video doesnt have the same tracks for gt and input")
                break
        
        ValImages = ValImages + sorted_OriginalsInTrack
        ValMasks =  ValMasks + sorted_MasksInTrack

print(len(ValImages))
print(len(ValMasks))  



Test = glob.glob(os.path.join(r'Small_ARIS_Mauzac_UnetReady\Test\2014*'))
TestImages = []
TestMasks = []

# Iterate over the provided image paths
for VideoPath in Test:

    TrackspathsMasks = glob.glob(os.path.join(VideoPath,"Foreground","t*"))
    TrackspathsOriginals = glob.glob(os.path.join(VideoPath,"Original","t*"))

    if len(TrackspathsMasks) != len(TrackspathsOriginals):
        print("Error : Video doesnt have the same tracks for gt and input")
        break

    for i in range(len(TrackspathsMasks)):

        MasksInTrack = glob.glob(os.path.join(TrackspathsMasks[i],"crop_*"))
        sorted_MasksInTrack = sorted(MasksInTrack, key=extract_frame_number)

        OriginalsInTrack = glob.glob(os.path.join(TrackspathsOriginals[i],"crop_*"))
        sorted_OriginalsInTrack = sorted(OriginalsInTrack, key=extract_frame_number)

        if len(sorted_MasksInTrack) != len(sorted_OriginalsInTrack):
                print("Error : Video doesnt have the same tracks for gt and input")
                break
        
        TestImages = TestImages + sorted_OriginalsInTrack
        TestMasks =  TestMasks + sorted_MasksInTrack

print(len(TestImages))
print(len(TestMasks))  

# Create datasets
train_dataset = CustomDataset(TrainImages, TrainMasks, transform=transform)
val_dataset = CustomDataset(ValImages, ValMasks, transform=transform)
#test_dataset = CustomDataset(test_image_paths, test_mask_paths, transform=transform)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
#test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# for VisualizationOnly, images, masks in train_loader:
#     images = images.to(device)  # Move images to GPU
#     masks = masks.to(device)    # Move masks to GPU
#     #print('wait')



model = UNet(n_class= 1)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
bce_loss = nn.BCEWithLogitsLoss()  # For binary segmentation

def dice_loss(inputs, targets):
    smooth = 1e-6
    inputs = torch.sigmoid(inputs)
    intersection = (inputs * targets).sum()
    union = inputs.sum() + targets.sum() + smooth
    dice = 1 - (2. * intersection + smooth) / union
    return dice

def dice_lossv1(y_true, y_pred, smooth=1e-5, thres=0.5):
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred >= thres).float()
    
    intersection = torch.sum(y_true * y_pred, dim=(1, 2, 3))
    sum_of_squares_pred = torch.sum(torch.square(y_pred), dim=(1, 2, 3))
    sum_of_squares_true = torch.sum(torch.square(y_true), dim=(1, 2, 3))
    
    dice_per_image = 1 - (2 * intersection + smooth) / (sum_of_squares_pred + sum_of_squares_true + smooth)
    
    # Calculate mean Dice loss across the batch
    dice_loss = 1 - torch.mean(dice_per_image)
    
    return dice_loss

num_epochs = 200
log_interval = 10
best_val_loss = float('inf')
patience = 10
early_stopping_counter = 0
VisualizationProb = 0.5

# Example training loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode

    for batch_idx, (VisualizationOnly, images, masks) in enumerate(train_loader):
        # Transfer data to GPU if available
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        bce = bce_loss(outputs, masks)
        dice = dice_loss(outputs, masks)
        #dice_lossv1
        #dice = dice_lossv1(outputs, masks)
        
        # Combine BCE and Dice Loss (adjust weight as needed)
        #loss = bce*0.5 + dice*0.5
        
        #loss = dice
        loss = dice

        # Zero gradients, perform backward pass, and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print training progress
        if batch_idx % log_interval*10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(images)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tBCE Loss: {bce.item():.6f}\tDice Loss: {dice.item():.6f}\tCombined Loss: {loss.item():.6f}')

            
        
        if batch_idx % log_interval == 0:
            
            visualize_sample(VisualizationOnly, masks, outputs, epoch, batch_idx, VisualizationProb, "Train")
    # Optionally, validate the model
    model.eval()  # Set model to evaluation mode for validation
    val_loss = 0.0
    with torch.no_grad():
        for VisualizationOnly_val, images_val, masks_val in val_loader:
            images_val = images_val.to(device)
            masks_val = masks_val.to(device)
            
            outputs_val = model(images_val)
            bce_val = bce_loss(outputs_val, masks_val)
            dice_val = dice_loss(outputs_val, masks_val)
            
            val_loss += (bce_val*0.5 + dice_val*0.5).item()

            visualize_sample(VisualizationOnly_val, masks_val, outputs_val, epoch, batch_idx, VisualizationProb, "Val")
    
    val_loss /= len(val_loader)
    print(f'Validation set: Average BCE+Dice Loss: {val_loss:.6f}')

    # Early stopping based on validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
        # Save the model when validation loss improves
        torch.save(model.state_dict(), 'unet_model.pth')
        print('Model saved.')
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print(f'Early stopping after {patience} epochs without improvement.')
            break

print('Training finished.')