import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


from utils.dice_score import jaccard2_coef


import matplotlib.patches as mpatches


def visualize_sample(images, masks, outputs, epoch, batch_idx, division, n_classes, session_folder):
    """
    To create the plot to compare the original iamge, the GT, the confidence prediciton mask, and the prediciton mask"""
    
    os.makedirs(os.path.join(str(session_folder), 'Visualize'), exist_ok=True)

    images = images.cpu().numpy()
    masks = F.one_hot(masks, n_classes).permute(0, 3, 1, 2).float()
    masks = masks.cpu().numpy()
    
    if outputs.shape[1] > 1:  # Multi-class case  # batch, 4, m, n     
        outputs = torch.softmax(outputs, dim=1) # batch, 4, m, n   
    
    
    RGB_PredImage = outputs.detach().cpu().numpy()
    # Apply threshold to outputs to get binary predictions for each class
    if outputs.shape[1] > 1:
        #outputs = (outputs >= VisualizationProb).detach().cpu().numpy() ## cambiar a argmax
        outputs = F.one_hot(torch.argmax(outputs, dim = 1), 4).permute(0, 1, 2, 3).float() # batch, m, n    now the values are the indices of the channel class, meaning the prediction, afte one hot
        outputs = outputs.detach().cpu().numpy()
    

    class_names = ["background","Trash", "SmallFish", "Eel"]  # Adjust as needed for your classes
    num_classes = len(class_names)
    
    colors = [
        [0, 0, 0],      # background - black
        [255, 0, 0],    # Trash - red
        [0, 255, 0],    # SmallFish - green
        [0, 0, 255],    # Eel - blue
    ]

    def create_combined_mask(mask):
        combined_mask = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)
        for i in range(num_classes):
            combined_mask[mask[i] == 1] = colors[i]
        return combined_mask
    
    
    xc = np.transpose(images[0], (1, 2, 0))
    
    if not np.shape(xc)[-1] == 3:
        # Iterate over each sample in the batch
        for idx in range(images.shape[0]):
            fig, axes = plt.subplots(1, 4, figsize=(15, 5))
            
            input_image = np.transpose(images[idx], (1, 2, 0))
    
            # Plot input image
            
            if np.shape(input_image)[-1] == 2:
                plot_input = np.zeros((np.shape(input_image)[0],np.shape(input_image)[1], 3))
                input_image[:,:,1] = input_image[:,:,1]*0.17
                plot_input[...,:2] = input_image
                input_image = plot_input
                axes[0].imshow((input_image* 255).astype(np.uint8))
                axes[0].set_title("Input Image")
                axes[0].axis('off')
            axes[0].imshow((input_image* 255).astype(np.uint8), cmap='gray')
            axes[0].set_title("Input Image")
            axes[0].axis('off')
    
            # Plot true mask
            true_mask_combined = create_combined_mask(masks[idx])
            print("The max value in true mask is: ", np.max(true_mask_combined))
            axes[1].imshow(true_mask_combined)
            axes[1].set_title("True Mask")
            axes[1].axis('off')
            
            # Plot confidence rgb image
            confidence_mask = RGB_PredImage[idx][1:]
            confidence_mask = np.transpose(confidence_mask, (1, 2, 0))
            axes[2].imshow((confidence_mask* 255).astype(np.uint8))
            axes[2].set_title("Confidence of Predicted Mask")
            axes[2].axis('off')
    
            #outputs
            # Plot predicted mask
            #pred_mask_combined = create_combined_mask(outputs[idx])
            axes[3].imshow(outputs[idx,:,:,1:])
            axes[3].set_title("Predicted Mask")
            axes[3].axis('off')
    
            # Create legend
            patches = [mpatches.Patch(color=np.array(color) / 255, label=class_name) for color, class_name in zip(colors, class_names)]
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    
            plt.tight_layout()
    
            # Save the figure for the current sample
            plt.savefig(os.path.join(session_folder, 'Visualize', f'{division}_visualization_epoch_{epoch}_batch_{batch_idx}_sample_{idx}.png'), bbox_inches='tight')
            plt.close()
    else:
        
        for idx in range(images.shape[0]):
            fig, axes = plt.subplots(1, 6, figsize=(15, 5))
    
            
            input_image = np.transpose(images[idx], (1, 2, 0))
            
            # Plot previous frame
            axes[0].imshow((input_image[:,:,0]* 255).astype(np.uint8), cmap='gray')
            axes[0].set_title("Previous Frame")
            axes[0].axis('off')
            print("The max value in previous frame is: ", np.max(input_image[:,:,0]))
    
            # Plot current frame
            axes[1].imshow((input_image[:,:,1]* 255).astype(np.uint8), cmap='gray')
            axes[1].set_title("Current Frame")
            axes[1].axis('off')
            print("The max value in Current frame is: ", np.max(input_image[:,:,1]))
    
            # Plot next frame
            axes[2].imshow((input_image[:,:,2]* 255).astype(np.uint8), cmap='gray')
            axes[2].set_title("Next Frame")
            axes[2].axis('off')
            print("The max value in Next frame is: ", np.max(input_image[:,:,2]))

            
            # Plot true mask
            true_mask_combined = create_combined_mask(masks[idx])
            print("The max value in true mask is: ", np.max(true_mask_combined))
            axes[3].imshow(true_mask_combined)
            axes[3].set_title("True Mask")
            axes[3].axis('off')
            
            # Plot confidence rgb image
            confidence_mask = (RGB_PredImage[idx][1:]* 255).astype(np.uint8)
            confidence_mask = np.transpose(confidence_mask, (1, 2, 0))
            axes[4].imshow(confidence_mask)
            axes[4].set_title("Confidence of Predicted Mask")
            axes[4].axis('off')
            print("The max value in confidence rgb mask is: ", np.max(confidence_mask))
            
            #outputs
            # Plot predicted mask
            #pred_mask_combined = create_combined_mask(outputs[idx])
            axes[5].imshow(outputs[idx,:,:,1:])
            axes[5].set_title("Predicted Mask")
            axes[5].axis('off')
            print("The max value in Predicted mask is: ", np.max(outputs[idx,:,:,1:]))
    
            # Create legend
            patches = [mpatches.Patch(color=np.array(color) / 255, label=class_name) for color, class_name in zip(colors, class_names)]
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    
            plt.tight_layout()
    
            # Save the figure for the current sample
            plt.savefig(os.path.join(session_folder, 'Visualize', f'{division}_visualization_epoch_{epoch}_batch_{batch_idx}_sample_{idx}.png'), bbox_inches='tight')
            plt.close()


  
def Visualize_Preds(net, dataloader, device, amp, epoch, session_folder, division = "val"):

    """Predictions of the model to generate the image plots from visualize_sample"""
    
    class_names = ["background", "Trash", "SmallFish", "Eel"]
    net.eval()
    num_val_batches = len(dataloader)
    
    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
            
        for batch_idx, batch in enumerate(tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=True)):
        
            image, mask_true = batch['image'], batch['mask']
            
            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)# logits
            
            #(images, masks, outputs, epoch, batch_idx, VisualizationProb, division, n_classes, session_folder)
            visualize_sample(image, mask_true, mask_pred, epoch, batch_idx, division, 4, session_folder)   





# For the model evaluation
@torch.inference_mode()
def evaluate(net, dataloader, device, amp, session_folder, division = "val"):

    """For the model evaluation,
    a really important aspect to mention is the division parameter, it doesn'determine the folder of the images. If you write "test" it will evaluate and also create a confusion matrix
    and classification report whic takes a lot of time, so that is why it isnt done in "val" while training.
    It can be probably be done in a more organized way.

    The session_folder is just the automatic folder name that is generated in the training .py"""

    class_names = ["background", "Trash", "SmallFish", "Eel"]
    net.eval()
    num_val_batches = len(dataloader)
    jaccard_score = 0
    count = 0

    # for confusion matrix and classification report
    all_preds = []
    all_targets = []
    
    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
            
        for batch_idx, batch in enumerate(tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=True)):
        
            image, mask_true = batch['image'], batch['mask']
            
            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)
 
            assert mask_true.min() >= 0 and mask_true.max() < 4, 'True mask indices should be in [0, n_classes['
            # convert to one-hot format
            
            mask_true = F.one_hot(mask_true, 4).permute(0, 3, 1, 2).float()
            #mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
            mask_pred = F.softmax(mask_pred, dim=1).float()  
            
            jaccard_score += jaccard2_coef( mask_true[:,1:], mask_pred[:,1:])#[:,1:]
            count += 1
                

                # Collect predictions and targets batch-wise
            if division =="test":
                all_preds = list(all_preds) + list(torch.argmax(mask_pred, dim =1).flatten().cpu().numpy())
                all_targets = list(all_targets) + list(torch.argmax(mask_true, dim =1).flatten().cpu().numpy())


                

    # Compute classification report
    if division =="test":
        class_report = classification_report(all_targets, all_preds, target_names=class_names, zero_division=0)
        
        
        print(class_report)
        cm = confusion_matrix(all_targets, all_preds)
    
        # Normalize the confusion matrix by row (i.e by the number of samples in each class)
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix (Percentage)')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # Save the plot as an image file
        plt.savefig(os.path.join(session_folder, 'confusion_matrix_percentage.png'))
        
        net.train()
        return jaccard_score / num_val_batches, class_report
        
    else:

        net.train()
        return jaccard_score / num_val_batches
    
    




            
            

 