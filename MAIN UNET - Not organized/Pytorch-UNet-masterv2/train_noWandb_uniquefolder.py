import argparse
import logging
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import inspect
import torchsummary

from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
import tempfile

from evaluate3 import evaluate
from unet import UNet
from utils.data_loading import BasicDataset
from utils.data_loading_bg import BGBasicDataset, add_speckle_noise, add_gaussian_noise, random_rotation, random_contrast
from utils.spaciotemporal_data_loading import TemporalBasicDataset
from utils.dice_score import jaccard2_coef, jaccard2_loss, adjust_softmax_predictions
from utils.utils import log_training_results

def write_to_file(folder_path, file_name, content):
    """
    Appends the given content to a file, creating the file and its parent folders if they don't exist.
    
    Parameters:
    - folder_path (str): The path to the folder where the file will be saved.
    - file_name (str): The name of the file to write to.
    - content (str): The content to append to the file.
    """
    os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist
    file_path = os.path.join(folder_path, file_name)  # Full path to the file

    with open(file_path, 'a') as file:  # Use 'a' mode to append content
        file.write(content + '\n')  # Add a newline after the content

    print(f'Content appended to {file_path}')

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=80, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=10, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default = 1e-5, help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--TrainRotation', action='store_true', default=False, help='Data Augmentation Rotation')
    parser.add_argument('--verticalFlip', action='store_true', default=False, help='Data Augmentation Vertical Flip')
    parser.add_argument('--contrast', action='store_true', default=False, help='Data Augmentation of Contrast')
    parser.add_argument('--TempDim', action='store_true', default=False, help='Temporal dimension: Past, current and next frame')
    parser.add_argument('--ModLoss', action='store_true', default=False, help='Modified loss')
    parser.add_argument('--BGS_input', action='store_true', default=False, help='Using BGS together with the input, 2D input')
    parser.add_argument('--GaussianNoise', action='store_true', default=False, help='Add Gaussian noise')
    parser.add_argument('--SpeckleNoise', action='store_true', default=False, help='Add Speckle noise')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0, help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--patience', '-p', type=int, default=10, help='Scheduler patience')
    parser.add_argument('--early_stopping_patience', '-ep', type=int, help='Early stopping patience')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=4, help='Number of classes')
    parser.add_argument('--weight_decay', '-w', type=float, default=4, help=' weight decay')
    parser.add_argument('--dataset_size', '-d', type=str, choices=['500x500', '750x664', '350x350', '1276x664'], default='350x350', help='Dataset size to use')
    return parser.parse_args()

def train_model(model, device, epochs=5, batch_size=5, learning_rate=1e-5, img_scale=1, rotation=False,
                verticalFlip=False, contrast=False, TempDim=False, amp=False, weight_decay=2.7549377795644327e-05, patience=100, early_stopping_patience=50, gradient_clipping=1.0,
                dataset_size='1276x664', ModLoss = False, BGS_input = False, GaussianNoise = False, SpeckleNoise = False, gaussian_var =  0.0001078916145171696, 
                                             speckle_factor =  0.29835877297444385, contrast_range = (0.6, 1.5), rot_degree = 15 ):
    start_time = time.time()
    if not TempDim:
        dir_img = Path(f'./{dataset_size}/data_train/imgs/')
        dir_mask = Path(f'./{dataset_size}/data_train/masks/')
    else: 
        dir_img = Path('FullSize1276x664/data_train/imgs/')
        dir_mask = Path('FullSize1276x664/data_train/masks/')
    if BGS_input:
        dir_bgs = Path(f'./{dataset_size}/data_train/bgs/')
        print("Training images loaded from: ", str(dir_img), ", ", str(dir_mask), ", ", str(dir_bgs))
    else:
        print("Training images loaded from: ", str(dir_img), ", ", str(dir_mask))
    if not TempDim:
        dir_img_val = Path(f'./{dataset_size}/data_val/imgs/')
        dir_mask_val = Path(f'./{dataset_size}/data_val/masks/')
    else:
        dir_img_val = Path('FullSize1276x664/data_val/imgs/')
        dir_mask_val = Path('FullSize1276x664/data_val/masks/')
        
    if BGS_input:
        dir_bgs_val = Path(f'./{dataset_size}/data_val/bgs/')
        print("Validation images loaded from: ", str(dir_img_val), ", ", str(dir_mask_val), ", ", str(dir_bgs_val))
    else:
        print("Validation images loaded from: ", str(dir_img_val), ", ", str(dir_mask_val))

    
    if TempDim:
        height = int(dataset_size.split("x")[0])
        width = int(dataset_size.split("x")[1])
        train_dataset = TemporalBasicDataset(dir_img, dir_mask, img_scale, mask_suffix="m_*_", rotation=rotation, vertical_flip=verticalFlip, contrast=contrast, 
                                             window_height = height, window_width = width, gaussian =GaussianNoise, speckle= SpeckleNoise, gaussian_var =  gaussian_var, 
                                             speckle_factor =  speckle_factor, contrast_range = contrast_range, rot_degree = rot_degree)
        val_dataset = TemporalBasicDataset(dir_img_val, dir_mask_val, img_scale, mask_suffix="m_*_", 
                                             window_height = height, window_width = width)
    elif BGS_input:
        train_dataset = BGBasicDataset(dir_img, dir_mask, dir_bgs, img_scale, mask_suffix="crop_m_*_", rotation=rotation, vertical_flip=verticalFlip, contrast=contrast, gaussian =GaussianNoise, speckle= SpeckleNoise, gaussian_var =  gaussian_var, 
                                             speckle_factor =  speckle_factor, contrast_range = contrast_range, rot_degree = rot_degree)
                                             
        val_dataset = BGBasicDataset(dir_img_val, dir_mask_val, dir_bgs_val, img_scale, mask_suffix="crop_m_*_")
    else:
        train_dataset = BasicDataset(dir_img, dir_mask, img_scale, mask_suffix="crop_m_*_", rotation=rotation, vertical_flip=verticalFlip, contrast=contrast, gaussian =GaussianNoise, speckle= SpeckleNoise, gaussian_var =  gaussian_var, 
        speckle_factor =  speckle_factor, contrast_range = contrast_range, rot_degree = rot_degree)
        
        val_dataset = BasicDataset(dir_img_val, dir_mask_val, img_scale, mask_suffix="crop_m_*_")

    n_val = len(val_dataset)
    n_train = len(train_dataset)


    loader_args = dict(batch_size=batch_size, num_workers=1, pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_args)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    session_folder = f'training_session_{timestamp}'
    session_dir = Path(session_folder)
    session_dir.mkdir(parents=True, exist_ok=True)

    #  logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s', handlers=[logging.FileHandler(session_dir / 'training.log'), logging.StreamHandler()])
   
    info = f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
        Patience:        {patience}
        E_stop_patience: {early_stopping_patience}
        Img Size:        {dataset_size}
        Temporal Dim:    {TempDim}
        Temporal Dim BG  {BGS_input}
        ModLoss:         {ModLoss}
        
       
        Rotation:        {rotation}
        Rotation Degrees: {rot_degree }

        VerticalFlip     {verticalFlip}

        Contrast:        {contrast}
        contrast range:  {contrast_range }

        Gaussian Noise:  {GaussianNoise}
        Gaussian var:    {gaussian_var }

        Speckle Noise:   {SpeckleNoise}
        Speckle factor:  {speckle_factor }
        FIlters:         

    '''
    #logging.info(info)
    print(info)
   
    file_name = 'ModelHyperparameters.txt'
    write_to_file(session_folder, file_name, info)

    train_losses = []
    val_scores = []
    train_scores = []

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=patience, min_lr=1e-6)
    print("Setting up scheduler patience at: ", patience)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    global_step = 0

    val_scores_np = []
    train_scores_np = []
   
    best_val_score = float('-inf')
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopping_patience = early_stopping_patience  # Set this to your desired patience level
    best_model_path = None
    best_model = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_train_score = 0
        epoch_val_score = 0
        val_count = 0
        train_count = 0
        loss = 0
       
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']
                #assert images.shape[1] == model.n_channels, f'Network has been defined with {model.n_channels} input channels, but loaded images have {images.shape[1]} channels. Please check that the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = model(images)
                       
                    input_softmax = F.softmax(masks_pred, dim=1).float() ##
                    target_one_hot = F.one_hot(true_masks, num_classes=4).permute(0, 3, 1, 2).float()##
                        
                       
                    if ModLoss:
                        input_softmax = adjust_softmax_predictions(input_softmax, target_one_hot)##
                        
                    loss = jaccard2_loss(target_one_hot[:,1:], input_softmax[:,1:]) ####
                       
                    
                    train_count += 1
                    

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])  
                global_step += 1
                epoch_loss += loss.item()
                epoch_train_score += 1 - loss.item()
                train_losses.append(loss.item())
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                   
        val_score = evaluate(net=model, dataloader=val_loader, device=device, amp=amp, session_folder=session_folder)
        scheduler.step(val_score)
        
        #tune.report(loss=epoch_loss / train_count, val_score=val_score)
                  
       
       
                   

        train_scores.append(epoch_train_score / train_count)
        
        val_scores.append(val_score)
       
        if val_score > best_val_score:
       
            best_val_score = val_score
            best_epoch = epoch
            
            model_path = session_dir / f'best_model_epoch{best_epoch}.pth'
            
            best_model_path = model_path
            best_model = model.state_dict()
            logging.info(f'New best model saved with validation score {best_val_score}')
            write_to_file(session_folder, "Training.txt", f'New best model saved with validation score {best_val_score}')
            
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            logging.info(f'Early stopping triggered after {early_stopping_patience} epochs without improvement')
            print(f'Early stopping triggered after {early_stopping_patience} epochs without improvement')
            write_to_file(session_folder, "Training.txt", f'Early stopping triggered after {early_stopping_patience} epochs without improvement')
            
            break
           
        logging.info(f'Epoch {epoch} Training Jaccard Score: {train_scores[-1]:.4f}')
        logging.info(f'Epoch {epoch} Validation Jaccard Score: {val_scores[-1]:.4f}')
        write_to_file(session_folder, "Training.txt", f'Epoch {epoch} Training Jaccard Score: {train_scores[-1]:.4f}')
        write_to_file(session_folder, "Training.txt", f'Epoch {epoch} Validation Jaccard Score: {val_scores[-1]:.4f}')
     
    for score in val_scores:
        val_scores_np.append(score.cpu().numpy())
       
    for score in train_scores:
        train_scores_np.append(score)
       
    plt.figure()
    plt.plot(np.arange(len(train_losses)), train_losses, label='Train Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig(session_dir / 'train_loss.png')

    plt.figure()
    plt.plot(np.arange(len(val_scores_np)), val_scores_np, label='Validation Jaccard Score')
    plt.plot(np.arange(len(train_scores_np)), train_scores_np, label='Training Jaccard Score')
    plt.xlabel('Epoch')
    plt.ylabel('Jaccard Score')
    plt.title('Jaccard Scores')
    plt.legend()
    plt.savefig(session_dir / 'jaccard_scores.png')
   
    if best_model_path:
        #logging.info(f'Best model path: {best_model_path}')
        write_to_file(session_folder, "Training.txt", f'Best model path: {best_model_path}')
    torch.save(best_model, best_model_path)
    print("Loading best weights to model to do confusion matrix at the end and classification report")
    write_to_file(session_folder, "Training.txt", "Loading best weights to model to do confusion matrix at the end and classification report")
    model.load_state_dict(best_model)
    val_score, class_report = evaluate(net = model, dataloader = val_loader, device = device,  amp = amp, session_folder = session_folder, division = "test") # Aca dice test, pero el datalodear es val, solo para que genere la matriz de confusion y el class report
    
    write_to_file(session_folder, "Training.txt", class_report)
    end_time = time.time()
    elapsed_time =  end_time - start_time
    write_to_file(session_folder, "Training.txt", f"Time passed in seconds is {elapsed_time}")
    
    session_data = {
    'Session_ID': f"{session_folder}",
    'Retrained from:': None,
    'LR (1e-)': learning_rate,
    'Patience': 10,
    'EarlyStop Patience': early_stopping_patience,
    'Epochs': epochs,
    'Best epoch': int(best_epoch),
    'batch_size': batch_size,
    'img_size': dataset_size,
    'Rotation': rotation,
    'V_Flip': verticalFlip,
    'Contrast': contrast,
    'Gaussian Noise': GaussianNoise,
    'Speckle Noise': SpeckleNoise,
    'Frame Stacking': TempDim,
    'BG': BGS_input,
    'MOD Loss': ModLoss,
    'Train': train_scores_np[best_epoch-1],
    'Val No BG': best_val_score.item(),
    'Test No BG': None,
    'Time': elapsed_time
}
    log_training_results('training_log_AllModels.csv', session_data)
    print(f"Best Validation Score: {best_val_score}")
    return best_model, best_val_score.item()

if __name__ == '__main__':
    start_time = time.time()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    
    filters = (64, 128, 256, 512, 1024)
    print(filters)
    if args.TempDim:
        model = UNet(n_channels=3, n_classes=args.classes, filters = filters)

    elif args.BGS_input:
        model = UNet(n_channels=2, n_classes=args.classes, filters = filters)

    else:
        model = UNet(n_channels=1, n_classes=args.classes, filters = filters)
        
    model = model.to(memory_format=torch.channels_last)
    torchsummary.summary(model)
    model.to(device=device)

    logging.info(f'Network:\n\t{model.n_channels} input channels\n\t{model.n_classes} output channels (classes)\n\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')
    print(f'Network:\n\t{model.n_channels} input channels\n\t{model.n_classes} output channels (classes)\n\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        
        if any(key.startswith('module.') for key in state_dict.keys()):
            # The model was saved using DataParallel, so we wrap the model with DataParallel
            model = torch.nn.DataParallel(model)
            # Remove the 'module.' prefix
            
            model.load_state_dict(state_dict)
        else:
            # The model was not saved using DataParallel, so we can load directly
            model.load_state_dict(state_dict)
        
        logging.info(f'Model loaded from {args.load}')
        print(f'Model loaded from {args.load}')

    

    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            rotation=args.TrainRotation,
            verticalFlip=args.verticalFlip,
            contrast=args.contrast,
            TempDim=args.TempDim,
            ModLoss = args.ModLoss,
            BGS_input=args.BGS_input,
            amp=args.amp,
            patience=args.patience,
            early_stopping_patience = args.early_stopping_patience,
            weight_decay = args.weight_decay,
            
            dataset_size=args.dataset_size,
            GaussianNoise =args.GaussianNoise,
            SpeckleNoise = args.SpeckleNoise
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! Enabling checkpointing to reduce memory usage, but this slows down training. Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            rotation=args.TrainRotation,
            verticalFlip=args.verticalFlip,
            contrast=args.contrast,
            TempDim=args.TempDim,
            amp=args.amp,
            patience=args.patience,
            
            dataset_size=args.dataset_size
        )
    end_time = time.time()
    elapsed_time =  end_time - start_time
        
    print("Time taken for the whole training code : ", str(elapsed_time))

