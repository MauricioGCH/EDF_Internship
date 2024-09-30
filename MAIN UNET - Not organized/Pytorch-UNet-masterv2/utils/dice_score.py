import torch
from torch import Tensor
import torchmetrics

  
        

def jaccard2_coef(y_true, y_pred, smooth=1e-6):
#    # Flatten the tensors
    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred.reshape(-1)
    
    # Calculate intersection and union
    intersection = torch.sum(y_true_f * y_pred_f)
    union = torch.sum(y_true_f * y_true_f) + torch.sum(y_pred_f * y_pred_f) - intersection
    
    # Compute the Jaccard coefficient
    return (intersection + smooth) / (union + smooth)

#def jaccard2_coef(y_true, y_pred, smooth=1e-6):
    # Move tensors to CPU for calculation
    #y_true_cpu = y_true.cpu()
    #y_pred_cpu = y_pred.cpu()

    # Flatten the tensors
    #y_true_f = y_true_cpu.reshape(-1)
    #y_pred_f = y_pred_cpu.reshape(-1)
    
    # Calculate intersection and union
    #intersection = torch.sum(y_true_f * y_pred_f)
    #union = torch.sum(y_true_f * y_true_f) + torch.sum(y_pred_f * y_pred_f) - intersection
    
    # Compute the Jaccard coefficient
    #return (intersection + smooth) / (union + smooth)

def jaccard2_loss(y_true, y_pred, smooth=1e-6):
    return 1 - jaccard2_coef(y_true, y_pred, smooth)


def adjust_softmax_predictions(input: Tensor, target: Tensor, epsilon: float = 1e-6) -> Tensor:
    # Clone the input tensor to avoid modifying the original
    adjusted_input = input.clone()
    
    # Check input dimensions
    assert input.dim() == 4, "Input should be a 4D tensor (NCHW)."
    assert target.dim() == 4, "Target should be a 4D tensor (NCHW)."
    assert input.size() == target.size(), "Input and target tensors must have the same shape."
    
    # Number of classes
    num_classes = input.size(1)
    
    # Create masks for the target classes based on integer class values
    # Unsqueeze to add a channel dimension and repeat along the channel dimension
    trash_target = (target[:, 1] == 1).float().unsqueeze(1).repeat(1, num_classes, 1, 1)
    background_target = (target[:, 0] == 1).float().unsqueeze(1).repeat(1, num_classes, 1, 1)
    
    # Ensure the target masks have the same shape as the input tensor
    assert trash_target.size() == input.size(), f"Trash target mask should have the same shape as the input. Got {trash_target.size()} and {input.size()}."
    assert background_target.size() == input.size(), f"Background target mask should have the same shape as the input. Got {background_target.size()} and {input.size()}."
    
    # Identify where the predictions are 'trash' or 'background'
    pred_trash = (input[:, 1] > 0.5).float().unsqueeze(1).repeat(1, num_classes, 1, 1)
    pred_background = (input[:, 0] > 0.5).float().unsqueeze(1).repeat(1, num_classes, 1, 1)
    
    # Ensure the prediction masks have the same shape as the input tensor
    assert pred_trash.size() == input.size(), f"Prediction trash mask should have the same shape as the input. Got {pred_trash.size()} and {input.size()}."
    assert pred_background.size() == input.size(), f"Prediction background mask should have the same shape as the input. Got {pred_background.size()} and {input.size()}."
    
    # Correct the prediction mistakes
    # Set the probability of 'trash' to 1 where the target is 'trash' and prediction is not 'trash'
    adjusted_input[:, 1] = torch.where(trash_target[:, 1].bool() & (pred_trash[:, 1] == 0), torch.tensor(1.0, device=input.device), adjusted_input[:, 1])
    
    # Set the probability of 'background' to 1 where the target is 'background' and prediction is not 'background'
    adjusted_input[:, 0] = torch.where(background_target[:, 0].bool() & (pred_background[:, 0] == 0), torch.tensor(1.0, device=input.device), adjusted_input[:, 0])
    
    # Ensure that after adjustments, no class probabilities are negative
    assert (adjusted_input >= 0).all(), "Adjusted input contains negative probabilities."
    
    # Create masks for adjusting probabilities of other classes (class 2 and class 3)
    # Only adjust these classes where the target is 'trash' or 'background'
    class_2_mask = trash_target[:, 1] + background_target[:, 0]  # Where target is either 'trash' or 'background'
    class_3_mask = trash_target[:, 1] + background_target[:, 0]  # Same mask for class 3

    # Set the probability of class 2 to 0 where the target is 'trash' or 'background'
    adjusted_input[:, 2] = torch.where(class_2_mask.bool(), torch.tensor(0.0, device=input.device), adjusted_input[:, 2])
    
    # Set the probability of class 3 to 0 where the target is 'trash' or 'background'
    adjusted_input[:, 3] = torch.where(class_3_mask.bool(), torch.tensor(0.0, device=input.device), adjusted_input[:, 3])

    # Ensure that after adjustments, no class probabilities are negative
    assert (adjusted_input >= 0).all(), "Adjusted input contains negative probabilities."
    
    # Ensure the sum of probabilities remains 1 for each pixel
    adjusted_input_sum = adjusted_input.sum(dim=1, keepdim=True)
    
    # Add a small epsilon to avoid division by zero
    assert (adjusted_input_sum > 0).all(), "Adjusted input sum for some pixels is zero, which could lead to division by zero."
    
    adjusted_input_sum = adjusted_input_sum + epsilon
    
    # Normalize the adjusted input so that the sum of probabilities is 1 for each pixel
    adjusted_input = adjusted_input / adjusted_input_sum
    
    # Ensure that after normalization, the sum of probabilities is 1 for each pixel
    assert torch.allclose(adjusted_input.sum(dim=1), torch.tensor(1.0, device=input.device), atol=epsilon), "Sum of probabilities for some pixels is not close to 1."
    
    return adjusted_input

