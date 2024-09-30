import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, auc

from utils.dice_score import multiclass_dice_coeff, dice_coeff, jaccard_coeff, multiclass_jaccard_coeff

import matplotlib.patches as mpatches

def visualize_sample(images, masks, outputs, epoch, batch_idx, VisualizationProb, division, n_classes, session_folder):
    os.makedirs(os.path.join(str(session_folder), 'Visualize'), exist_ok=True)

    images = images.cpu().numpy()
    masks = F.one_hot(masks, n_classes).permute(0, 3, 1, 2).float()
    masks = masks.cpu().numpy()
    
    if outputs.shape[1] > 1:  # Multi-class case
        outputs = torch.softmax(outputs, dim=1)
    else:  # Binary case
        outputs = torch.sigmoid(outputs)
    
    VisualizationProb = 0.5  # Adjust this threshold as needed

    # Apply threshold to outputs to get binary predictions for each class
    if outputs.shape[1] > 1:
        outputs = (outputs >= VisualizationProb).detach().cpu().numpy()
    else:
        outputs = (outputs.squeeze(1) >= VisualizationProb).detach().cpu().numpy()

    class_names = ["background","Trash", "SmallFish", "Eel"]
    num_classes = len(class_names)
    
    colors = [
        [0, 0, 0],        # background - black
        [255, 0, 0],      # Trash - red
        [0, 255, 0],      # SmallFish - green
        [0, 0, 255],      #  Eel- blue
        [255, 255, 0],    # Silure - yellow
        [255, 0, 255],    # salom atlantic - magenta
    ]
#F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float()
    def create_combined_mask(mask):
        combined_mask = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)
        for i in range(num_classes):
            combined_mask[mask[i] == 1] = colors[i]
        return combined_mask

    if division == "val":
    
        fig, axes = plt.subplots(3, int(images.shape[0]/2), figsize=(15, 15))
        for i in range(int(images.shape[0]/2)):
            # Plot input image
            axes[0, i].imshow(np.transpose(images[i], (1, 2, 0)), cmap = "gray")
            axes[0, i].set_title("Input Image")
            axes[0, i].axis('off')

            # Plot true mask
            true_mask_combined = create_combined_mask(masks[i])
            axes[1, i].imshow(true_mask_combined)
            axes[1, i].set_title("True Mask")
            axes[1, i].axis('off')

            # Plot predicted mask
            pred_mask_combined = create_combined_mask(outputs[i])
            axes[2, i].imshow(pred_mask_combined)
            axes[2, i].set_title("Predicted Mask")
            axes[2, i].axis('off')

        # Create legend
        patches = [mpatches.Patch(color=np.array(color) / 255, label=class_name) for color, class_name in zip(colors, class_names)]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(os.path.join(session_folder,'Visualize', f'{division}_visualization_epoch_{epoch}_batch_{batch_idx}.png'), bbox_inches='tight')
        plt.close()



@torch.inference_mode()
def evaluate(net, dataloader, device, amp, epoch,session_folder, log_interval=10, mask_threshold=0.5):
    class_names = ["background","Trash_Arcing", "SmallFish_Arcing", "Eel_Arcing", "SmallFish", "Trash", "Eel"]
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    all_preds = []
    all_targets = []

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch_idx, batch in enumerate(tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False)):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)
            #probs = F.sigmoid(mask_pred).detach().cpu().numpy()
            #masks = mask_true.detach().cpu().numpy()
            
            # Collect all predictions and targets for precision-recall curve

            #all_preds.extend(probs.flatten())
            #all_targets.extend(masks.flatten())

            #def visualize_sample(images, masks, outputs, epoch, batch_idx, VisualizationProb, division):

            #if batch_idx % log_interval == 0:
            visualize_sample(image, mask_true, mask_pred, epoch, batch_idx, mask_threshold, "val", n_classes= net.n_classes, session_folder = session_folder)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > mask_threshold).float()
                # compute the Dice score
                dice_score += jaccard_coeff(mask_pred.squeeze(1), mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_jaccard_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    # Compute precision-recall curve and AUC
    #precision, recall, thresholds = precision_recall_curve(all_targets, all_preds)
    """ pr_auc = auc(recall, precision)

    # Plot and save precision-recall curve
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AUC = {pr_auc:.2f})')
    plt.savefig(f'precision_recall_curve_epoch_{epoch}.png')
    plt.close() """

    net.train()
    return dice_score / max(num_val_batches, 1)