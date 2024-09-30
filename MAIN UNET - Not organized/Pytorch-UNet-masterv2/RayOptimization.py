import os
import sys
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import psutil

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
from ray.tune import Callback

from evaluate3 import evaluate
from unet import UNet
from utils.data_loading import BasicDataset
from utils.data_loading_bg import BGBasicDataset, add_speckle_noise, add_gaussian_noise, random_rotation, random_contrast
from utils.spaciotemporal_data_loading import TemporalBasicDataset
from utils.dice_score import jaccard2_coef, jaccard2_loss
from pathlib import Path

os.environ["RAY_memory_usage_threshold"] = "0.90"




def train_ray(config):

    script_dir = Path("/fscronos/home/d42684/Documents/CODE/Pytorch-UNet-masterv2")
    if not config["TempDim"]:
        dir_img = script_dir / f'{config["dataset_size"]}/data_train/imgs/'
        dir_mask = script_dir / f'{config["dataset_size"]}/data_train/masks/'
    else: 
        dir_img = script_dir /'FullSize1276x664/data_train/imgs/'
        dir_mask = script_dir /'FullSize1276x664/data_train/masks/'
    if config["BGS_input"]:
        dir_bgs = script_dir /f'{config["dataset_size"]}/data_train/bgs/'
        print("Training images loaded from: ", str(dir_img), ", ", str(dir_mask), ", ", str(dir_bgs))
    else:
        print("Training images loaded from: ", str(dir_img), ", ", str(dir_mask))
    
    if not config["TempDim"]:
        dir_img_val = script_dir /f'{config["dataset_size"]}/data_val/imgs/'
        dir_mask_val = script_dir /f'{config["dataset_size"]}/data_val/masks/'
    else:
        dir_img_val = script_dir /'FullSize1276x664/data_val/imgs/'
        dir_mask_val = script_dir /'FullSize1276x664/data_val/masks/'
        
    if config["BGS_input"]:
        dir_bgs_val = script_dir /f'{config["dataset_size"]}/data_val/bgs/'
        print("Validation images loaded from: ", str(dir_img_val), ", ", str(dir_mask_val), ", ", str(dir_bgs_val))
    else:
        print("Validation images loaded from: ", str(dir_img_val), ", ", str(dir_mask_val))

    img_scale = config["img_scale"]
    rotation =  config["rotation"]
    verticalFlip = config["verticalFlip"]
    contrast = config["contrast"]
    GaussianNoise = config["GaussianNoise"]
    SpeckleNoise = config["SpeckleNoise"]

    if config["TempDim"]:
        height = int(config["dataset_size"].split("x")[0])
        width = int(config["dataset_size"].split("x")[1])
        train_dataset = TemporalBasicDataset(dir_img, dir_mask, img_scale, mask_suffix="m_*_", rotation=rotation, vertical_flip=verticalFlip, contrast=contrast, 
                                             window_height = height, window_width = width, gaussian =GaussianNoise, speckle= SpeckleNoise)
        val_dataset = TemporalBasicDataset(dir_img_val, dir_mask_val, img_scale, mask_suffix="m_*_", 
                                             window_height = height, window_width = width)
    elif config["BGS_input"]:
        train_dataset = BGBasicDataset(dir_img, dir_mask, dir_bgs, img_scale, mask_suffix="crop_m_*_", rotation=rotation, vertical_flip=verticalFlip, contrast=contrast, gaussian =GaussianNoise, speckle= SpeckleNoise)
        val_dataset = BGBasicDataset(dir_img_val, dir_mask_val, dir_bgs_val, img_scale, mask_suffix="crop_m_*_")
    else:
        train_dataset = BasicDataset(dir_img, dir_mask, img_scale, mask_suffix="crop_m_*_", rotation=rotation, vertical_flip=verticalFlip, contrast=contrast, gaussian =GaussianNoise, speckle= SpeckleNoise)
        val_dataset = BasicDataset(dir_img_val, dir_mask_val, img_scale, mask_suffix="crop_m_*_")

    n_val = len(val_dataset)
    n_train = len(train_dataset)

    loader_args = dict(batch_size=config["batch_size"], num_workers=1, pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_args)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    n_channels = 1
    if config["TempDim"]:
        n_channels = 3
    elif config["BGS_input"]:
        n_channels = 2
    filters = config["filters"]
    model = UNet( n_channels = n_channels, n_classes=4, filters = filters,  bilinear=False)
    model.to(device=device)


    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=config["patience"], min_lr=1e-6)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=config["amp"])
    global_step = 0
    
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        epoch_loss = 0
        epoch_train_score = 0
        train_count = 0
        jaccard_score = 0
        epochs = config["epochs"]
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=config["amp"]):

                    masks_pred = model(images)

                    pred_softmax = F.softmax(masks_pred, dim=1).float() ##
                    target_one_hot = F.one_hot(true_masks, num_classes=model.n_classes).permute(0, 3, 1, 2).float()##

                    loss = jaccard2_loss(target_one_hot, pred_softmax)
                    train_count += 1


                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clipping"])
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])  
                global_step += 1
                epoch_loss += loss.item()
                epoch_train_score += 1 - loss.item()
                
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        epoch_loss = epoch_loss/train_count #######
        model.eval()
        num_val_batches = len(val_loader)
        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=config["amp"]):
        
            for batch_idx, batch in enumerate(tqdm(val_loader, total=num_val_batches, desc='Validation round', unit='batch', leave=True)):
            
                image, mask_true = batch['image'], batch['mask']
                
                # move images and labels to correct device and type
                image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                mask_true = mask_true.to(device=device, dtype=torch.long)

                # predict the mask
                mask_pred = model(image)
                
                assert mask_true.min() >= 0 and mask_true.max() < 4, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, 4).permute(0, 3, 1, 2).float()
                #mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.softmax(mask_pred, dim=1).float()  
                
                jaccard_score += jaccard2_coef( mask_true[:,1:], mask_pred[:,1:])

        score = jaccard_score / max(num_val_batches, 1)
        scheduler.step(score)
        ray.train.report(dict(val_score=score.item()))
        #tune.report(val_score=score)
    

search_space = {
    "epochs": tune.choice([100]),
    "batch_size": tune.choice([8, 16]),
    "learning_rate": tune.loguniform(1e-6, 1e-4),
    "img_scale": tune.choice([0.5]),
    "rotation": tune.choice([True, False]),
    "verticalFlip": tune.choice([True, False]),
    "contrast": tune.choice([True, False]),
    "TempDim": tune.choice([False]),
    "amp": tune.choice([True]),
    "weight_decay": tune.loguniform(1e-8, 1e-4),
    "patience": tune.choice([10, 15, 20]),
    #"early_stopping_patience": tune.choice([20, 25, 30]),
    "gradient_clipping": tune.uniform(0.5, 2.0),
    #"ModLoss": tune.choice([True, False]),
    "BGS_input": tune.choice([False]),
    "GaussianNoise": tune.choice([True, False]),
    "SpeckleNoise": tune.choice([True, False]),
    "dataset_size": tune.choice(["350x350"]), #"500x500"
    "filters": tune.choice([[32, 64, 128, 256, 512]])#, [64, 128, 256, 512, 1024]
    }
def tune_func(config):
    tune.utils.wait_for_gpu()
    train_ray(config)


class MemoryMonitorCallback(Callback):
    def on_trial_result(self, iteration, trials, trial, result, **info):
        # Log CPU memory usage
        print(f"CPU Memory Usage: {psutil.virtual_memory().used / 1024 ** 2} MB")
        
        # Log GPU memory usage
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024 ** 2} MB")

if __name__ == "__main__":
    scheduler = ASHAScheduler(
    max_t=100,  # Maximum number of epochs
    grace_period=20,
    reduction_factor=3
    )
    #os.chdir(os.path.dirname(os.path.abspath(__file__)))
    #output_dir = os.getcwd()
    #algo = OptunaSearch()
    #algo = ConcurrencyLimiter(algo, max_concurrent=1)
    
    ray.init()
    print(ray.available_resources())
    analysis = tune.run(
        train_ray,
        config=search_space,
        num_samples=1,  # Adjust the number of trials
        scheduler=scheduler,
        metric="val_score",
        mode="max",
        resources_per_trial={"cpu": 10, "gpu": 4},
        callbacks=[MemoryMonitorCallback()]
        #search_alg=algo
        #local_dir=output_dir
    )

    print("Best hyperparameters found were: ", analysis.best_config)