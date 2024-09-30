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
import torchsummary

import inspect
from utils.data_loading import add_speckle_noise, add_gaussian_noise, random_rotation, random_contrast
from evaluate3 import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, TemporalBasicDataset, BGBasicDataset
from utils.dice_score import jaccard_loss, jaccard_coeff, adjust_softmax_predictions

def write_to_file(folder_path, file_name, content):
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'w') as file:
        file.write(content)
    print(f'Content written to {file_path}')

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=10, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5, help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--TrainRotation', action='store_true', default=False, help='Data Augmentation Rotation')
    parser.add_argument('--verticalFlip', action='store_true', default=False, help='Data Augmentation Vertical Flip')
    parser.add_argument('--contrast', action='store_true', default=False, help='Data Augmentation of Contrast')
    parser.add_argument('--TempDim', action='store_true', default=False, help='Temporal dimension: Past, current and next frame')
    parser.add_argument('--ModLoss', action='store_true', default=False, help='Modified loss')
    parser.add_argument('--GaussianNoise', action='store_true', default=False, help='Add Gaussian noise')
    parser.add_argument('--SpeckleNoise', action='store_true', default=False, help='Add Speckle noise')
    parser.add_argument('--BGS_input', action='store_true', default=False, help='Using BGS together with the input, 2D input')
    parser.add_argument('--patience', '-p', type=int, default=5, help='Scheduler patience')
    parser.add_argument('--early_stopping_patience', '-ep', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=4, help='Number of classes')
    parser.add_argument('--dataset_size', '-d', type=int, nargs=2, default=[350,350], help='Dataset size to use')
    return parser.parse_args()

def train_model(model, device, epochs=20, batch_size=5, learning_rate=1e-5, save_checkpoint=True, img_scale=1, rotation=False, 
                verticalFlip=False, contrast=False, TempDim=False, amp=False, weight_decay=1e-8, patience=100, early_stopping_patience=10, 
                gradient_clipping=1.0, dataset_size=[350,350], ModLoss = False, BGS_input = False, GaussianNoise = False, SpeckleNoise = False):
    
    dir_img = Path('FullSize1276x664/data_train/imgs/')
    dir_mask = Path('FullSize1276x664/data_train/masks/')
    if BGS_input:
        dir_bgs = Path('FullSize1276x664/data_train/bgs/')
        print("Training images loaded from: ", str(dir_img), ", ", str(dir_mask), ", ", str(dir_bgs))
    else:
        print("Training images loaded from: ", str(dir_img), ", ", str(dir_mask))
    
    dir_img_val = Path('FullSize1276x664/data_val/imgs/')
    dir_mask_val = Path('FullSize1276x664/data_val/masks/')
    if BGS_input:
        dir_bgs_val = Path('FullSize1276x664/data_val/bgs/')
        print("Validation images loaded from: ", str(dir_img_val), ", ", str(dir_mask_val), ", ", str(dir_bgs_val))
    else:
        print("Validation images loaded from: ", str(dir_img_val), ", ", str(dir_mask_val))

    if TempDim:
        train_dataset = TemporalBasicDataset(dir_img, dir_mask, img_scale, mask_suffix="m_*_", rotation=rotation, vertical_flip=verticalFlip, contrast=contrast, 
                                             window_height = dataset_size[0], window_width = dataset_size[1], gaussian =GaussianNoise, speckle= SpeckleNoise)
        val_dataset = TemporalBasicDataset(dir_img_val, dir_mask_val, img_scale, mask_suffix="m_*_", rotation=False, vertical_flip=False, window_height = dataset_size[0], window_width = dataset_size[1])
    elif BGS_input: # , gaussian =GaussianNoise, speckle= SpeckleNoise
        train_dataset = BGBasicDataset(dir_img, dir_mask, dir_bgs, img_scale, mask_suffix="m_*_", rotation=rotation, vertical_flip=verticalFlip, contrast=contrast, 
                                       window_height = dataset_size[0], window_width = dataset_size[1], gaussian =GaussianNoise, speckle= SpeckleNoise)
        val_dataset = BGBasicDataset(dir_img_val, dir_mask_val, dir_bgs_val, img_scale, mask_suffix="m_*_", rotation=False, vertical_flip=False, window_height = dataset_size[0], window_width = dataset_size[1])
    else:
        train_dataset = BasicDataset(dir_img, dir_mask, img_scale, mask_suffix="m_*_", rotation=rotation, vertical_flip=verticalFlip, contrast=contrast, 
                                     window_height = dataset_size[0], window_width = dataset_size[1], gaussian =GaussianNoise, speckle= SpeckleNoise)
        val_dataset = BasicDataset(dir_img_val, dir_mask_val, img_scale, mask_suffix="m_*_", rotation=False, vertical_flip=False, window_height = dataset_size[0], window_width = dataset_size[1])

    n_val = len(val_dataset)
    n_train = len(train_dataset)

    loader_args = dict(batch_size=batch_size, num_workers=1, pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=True, drop_last=True, **loader_args)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    session_folder = f'training_session_{timestamp}'
    session_dir = Path(session_folder)
    session_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s', handlers=[logging.FileHandler(session_dir / 'training.log'), logging.StreamHandler()])
    
    signaturegauss = inspect.signature(add_gaussian_noise)
    default_var = signaturegauss.parameters['var'].default

    signaturespec = inspect.signature(add_speckle_noise)
    default_factor = signaturespec.parameters['noise_factor'].default

    signaturerot = inspect.signature(random_rotation)
    default_angle = signaturerot.parameters['degrees'].default
   

    signaturerot = inspect.signature(random_contrast)
    default_CFactor = signaturerot.parameters['contrast_range'].default

    info = f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
        LR Patience:     {patience}
        Early stop Patience: {early_stopping_patience}
        Img Size:        {dataset_size}
        Temporal Dim:    {TempDim}
        Temporal Dim BG: {BGS_input}
        ModLoss:         {ModLoss}

        Rotation:        {rotation}
        Rotation Degrees:{default_angle}

        VerticalFlip     {verticalFlip}

        Contrast:        {contrast}
        contrast range:  {default_CFactor}

        Gaussian Noise:  {GaussianNoise}
        Gaussian var:    {default_var}

        Speckle Noise:   {SpeckleNoise}
        Speckle factor:  {default_factor}

        
        

    '''
    logging.info(info)
    print(info)
    file_name = 'ModelHyperparameters.txt'
    write_to_file(session_folder, file_name, info)

    train_losses = []
    val_scores = []
    train_scores = []

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=patience, min_lr=1e-6)
    print("Setting up scheduler patience at:", patience)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    global_step = 0

    val_scores_np = []
    train_scores_np = []

    best_val_score = float('-inf')
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopping_patience = 10  # Set this to your desired patience level
    best_model_path = None
    best_model = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_train_score = 0
        epoch_val_score = 0
        train_count = 0
        val_count = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']
                assert images.shape[1] == model.n_channels, f'Network has been defined with {model.n_channels} input channels, but loaded images have {images.shape[1]} channels. Please check that the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)
        
                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = jaccard_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                        train_score = jaccard_coeff(F.sigmoid(masks_pred.squeeze(1)), true_masks.float())
                    else:
                        
                        input_softmax = F.softmax(masks_pred, dim=1).float() ##
                        target_one_hot = F.one_hot(true_masks, num_classes=model.n_classes).permute(0, 3, 1, 2).float()##
                        
                        if ModLoss:
                            input_softmax = adjust_softmax_predictions(input_softmax, target_one_hot)##
                        
                        
                        loss = jaccard_loss(input_softmax, target_one_hot, multiclass=True) ####
                        train_count +=1
                        train_score = 1 - loss.item()
                    
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                epoch_train_score += train_score
                train_losses.append(loss.item())
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Validation at intervals
                division_step = (n_train // (5 * batch_size))
                if division_step > 0 and global_step % division_step == 0:
                    val_score = evaluate(net=model, dataloader=val_loader, device=device, amp=amp, session_folder=session_folder, ModLoss=ModLoss)
                    epoch_val_score += val_score
                    val_count += 1
                    logging.info('Validation Jaccard score at step {}: {:.4f}'.format(global_step, val_score))

        # Average validation score for the epoch
        avg_val_score = epoch_val_score / val_count if val_count > 0 else 0.0
        val_scores.append(avg_val_score)
        scheduler.step(avg_val_score)
        logging.info(f'Epoch {epoch} Average Validation Jaccard Score: {avg_val_score:.4f}')

        if avg_val_score > best_val_score:
            best_val_score = avg_val_score
            best_epoch = epoch
            if save_checkpoint:
                model_path = session_dir / f'best_model_epoch{best_epoch}.pth'
                torch.save(model.state_dict(), model_path)
                best_model_path = model_path
                best_model = model.state_dict()
                logging.info(f'New best model saved with validation score {best_val_score}')
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            logging.info(f'Early stopping triggered after {early_stopping_patience} epochs without improvement')
            break

        logging.info(f'Epoch {epoch} Training Jaccard Score: {train_scores[-1]:.4f}')
        logging.info(f'Epoch {epoch} Average Validation Jaccard Score: {val_scores[-1]:.4f}')

    for score in val_scores:
        val_scores_np.append(score)

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

    # Save the final best model path for reference
    if best_model_path:
        logging.info(f'Best model path: {best_model_path}')
    
    return best_model  # Return the best model state dictionary

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    if args.TempDim:
        model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
        torchsummary.summary(model, input_size= (3, args.dataset_size[0],args.dataset_size[1]))

    elif args.BGS_input:
        model = UNet(n_channels=2, n_classes=args.classes, bilinear=args.bilinear)
        torchsummary.summary(model, input_size= (2, args.dataset_size[0],args.dataset_size[1]))

    else:
        model = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
        torchsummary.summary(model, input_size= (1, args.dataset_size[0],args.dataset_size[1]))
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n\t{model.n_channels} input channels\n\t{model.n_classes} output channels (classes)\n\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')
    print(f'Network:\n\t{model.n_channels} input channels\n\t{model.n_classes} output channels (classes)\n\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')
    print("Model Architecture : ")

    
    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')
        print(f'Model loaded from {args.load}')

    model.to(device=device)
    torchsummary.summary(model, input_size=(model.n_channels, int(args.dataset_size[0]), int(args.dataset_size[1])))
#int(args.dataset_size[0]), int(args.dataset_size[1])
    # if args.TempDim:
    #     torchsummary.summary(model, input_size= (3, int(args.dataset_size[0]), int(args.dataset_size[1])))
    # elif args.BGS_input:
    #     torchsummary.summary(model, input_size= (2, int(args.dataset_size[0]), int(args.dataset_size[1])))
    # else:
    #     torchsummary.summary(model, input_size= (1, int(args.dataset_size[0]), int(args.dataset_size[1])))


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
            ModLoss=args.ModLoss,
            BGS_input=args.BGS_input,
            amp=args.amp,
            patience=args.patience,
            save_checkpoint=True,
            dataset_size=args.dataset_size,
            early_stopping_patience= args.early_stopping_patience
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
            save_checkpoint=True,
            dataset_size=args.dataset_size
        )