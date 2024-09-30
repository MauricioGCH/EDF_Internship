# HYperparameter search
from evaluate3 import evaluate
from unet import UNet
from utils.data_loading import BasicDataset
from utils.data_loading_bg import BGBasicDataset, add_speckle_noise, add_gaussian_noise, random_rotation, random_contrast
from utils.spaciotemporal_data_loading import TemporalBasicDataset
from utils.dice_score import jaccard2_coef, jaccard2_loss, adjust_softmax_predictions
from utils.utils import log_training_results
from NewTrain import train_model


import time
import os
import logging
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from ray import tune
import ray
from ray.tune.schedulers import ASHAScheduler




def train_model_tune(config):
    if config["BGS_input"] and config["TempDim"]:
        raise ValueError("BGS_input and TempDim cannot both be True.")
    
    device = torch.device('cuda')
    n_channels = 1
    if config["TempDim"]:
        n_channels = 3
    elif config["BGS_input"]:
        n_channels = 2
    filters = config["filters"]
    model = UNet( n_channels = n_channels, n_classes=4, filters = filters,  bilinear=False)
    model.to(device=device)
    
    best_model, best_val_score = train_model(
        model=model,
        device=device,
        #epochs=config.get("epochs", 5),
        batch_size=config.get("batch_size", 5),
        learning_rate=config.get("learning_rate", 1e-5),
        img_scale=config.get("img_scale", 1),
        rotation=config.get("rotation", False),
        verticalFlip=config.get("verticalFlip", False),
        contrast=config.get("contrast", False),
        TempDim=config.get("TempDim", False),
        amp=config.get("amp", False),
        weight_decay=config.get("weight_decay", 1e-8),
        patience=config.get("patience", 100),
        #early_stopping_patience=config.get("early_stopping_patience", 10),
        gradient_clipping=config.get("gradient_clipping", 1.0),
        #ModLoss=config.get("ModLoss", False),
        BGS_input=config.get("BGS_input", False),
        GaussianNoise=config.get("GaussianNoise", False),
        SpeckleNoise=config.get("SpeckleNoise", False)
    )
    
    #tune.report(val_score=best_val_score)

search_space = {
    #"epochs": tune.choice([100]),
    "batch_size": tune.choice([8, 16, 32]),
    "learning_rate": tune.loguniform(1e-6, 1e-4),
    "img_scale": tune.choice([0.5, 1.0]),
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
    "filters": tune.choice([[32, 64, 128, 256, 512], [64, 128, 256, 512, 1024]])
}


if __name__ == "__main__":
    scheduler = ASHAScheduler(
    max_t=100,  # Maximum number of epochs
    grace_period=20,
    reduction_factor=3
    )
    #os.chdir(os.path.dirname(os.path.abspath(__file__)))
    #output_dir = os.getcwd()
    ray.init()
    print(ray.available_resources())
    analysis = tune.run(
        train_model_tune,
        config=search_space,
        num_samples=10,  # Adjust the number of trials
        scheduler=scheduler,
        metric="val_score",
        mode="max",
        resources_per_trial={"cpu": 2, "gpu": 4}
        #local_dir=output_dir
    )

    print("Best hyperparameters found were: ", analysis.best_config)
