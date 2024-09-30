import torch

"""THe file says dice score, but it is for the jaccard2 score. I just didn't want to change it in all the other .py that it is called."""
        

def jaccard2_coef(y_true, y_pred, smooth=1e-6):
#    # Flatten the tensors
    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred.reshape(-1)
    
    # Calculate intersection and union
    intersection = torch.sum(y_true_f * y_pred_f)
    union = torch.sum(y_true_f * y_true_f) + torch.sum(y_pred_f * y_pred_f) - intersection
    
    # Compute the Jaccard coefficient
    return (intersection + smooth) / (union + smooth)


def jaccard2_loss(y_true, y_pred, smooth=1e-6):
    return 1 - jaccard2_coef(y_true, y_pred, smooth)


