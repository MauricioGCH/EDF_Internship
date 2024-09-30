import os
import shutil
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import optuna
from optuna.importance import FanovaImportanceEvaluator

from evaluate3 import evaluate
from unet import UNet
from utils.data_loading import BasicDataset
from utils.data_loading_bg import BGBasicDataset
from utils.spaciotemporal_data_loading import TemporalBasicDataset
from utils.dice_score import jaccard2_coef, jaccard2_loss
from pathlib import Path
from optuna.trial import TrialState
import time
import matplotlib.pyplot as plt
from test1 import test_model

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

# Define the objective function for Optuna
def objective(trial):
    # Define the hyperparameters to tune
    
    
    config = {
        "epochs": trial.suggest_int("epochs", 80, 80),  # Fixed to 80
        "batch_size": trial.suggest_categorical("batch_size", [16,10]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True), # trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "gradient_clipping": trial.suggest_float("gradient_clipping", 1, 1),  # Fixed value
        "patience": trial.suggest_categorical("patience", [10, 20]),  # Fixed value
        "filters": trial.suggest_categorical("filters", [(64, 128, 256, 512, 1024)]),
        "dataset_size": trial.suggest_categorical("dataset_size", ["350x350"]),
        "rotation": trial.suggest_categorical("rotation", [False]),
        "contrast": trial.suggest_categorical("contrast", [True]),
        "GaussianNoise": trial.suggest_categorical("GaussianNoise", [False]),
        "SpeckleNoise": trial.suggest_categorical("SpeckleNoise", [True]),
        "BGS_input": trial.suggest_categorical("BGS_input", [False]),  # Fixed value
        "TempDim": trial.suggest_categorical("TempDim", [False]),  # Fixed value
        "amp": trial.suggest_categorical("amp", [True]),  # Fixed value
         "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-8, log=True),
        "verticalFlip": trial.suggest_categorical("verticalFlip", [True]),  # Fixed value
        "img_scale": trial.suggest_categorical("img_scale", [1]),  # Fixed value
    }
    
    # Conditionally suggest parameters based on the values of other parameters
    if config["GaussianNoise"]:
        config["gaussian_var"] = trial.suggest_float("Gaussian_var", 2e-4, 2e-3, log=True)
    
    if config["SpeckleNoise"]:
        config["speckle_factor"] = trial.suggest_float("Speckle_factor", 0.05, 0.25)
    
    if config["rotation"]:
        config["rotation_degrees"] = trial.suggest_int("rotation_degrees", 10, 15 )
    
    if config["contrast"]:
        config["contrast_range"] = trial.suggest_categorical("contrast_range", [(0.6, 1.5), (0.6, 1.2)])
        
        

    # Paths and dataset preparation (same as before)
    script_dir = Path("/fscronos/home/d42684/Documents/CODE/Pytorch-UNet-masterv2")
    if not config["TempDim"]:
        dir_img = script_dir / f'{config["dataset_size"]}/data_train/imgs/'
        dir_mask = script_dir / f'{config["dataset_size"]}/data_train/masks/'
    else: 
        dir_img = script_dir /'FullSize1276x664/data_train/imgs/'
        dir_mask = script_dir /'FullSize1276x664/data_train/masks/'
        
    if config["BGS_input"]:
        dir_bgs = script_dir /f'{config["dataset_size"]}/data_train/bgs/'
    
    if not config["TempDim"]:
        dir_img_val = script_dir /f'{config["dataset_size"]}/data_val/imgs/'
        dir_mask_val = script_dir /f'{config["dataset_size"]}/data_val/masks/'
    else:
        dir_img_val = script_dir /'FullSize1276x664/data_val/imgs/'
        dir_mask_val = script_dir /'FullSize1276x664/data_val/masks/'
        
    if config["BGS_input"]:
        dir_bgs_val = script_dir /f'{config["dataset_size"]}/data_val/bgs/'

    img_scale = config["img_scale"]
    rotation =  config["rotation"]
    verticalFlip = config["verticalFlip"]
    contrast = config["contrast"]
    GaussianNoise = config["GaussianNoise"]
    SpeckleNoise = config["SpeckleNoise"]
    
    if GaussianNoise:
        gaussian_var = config["gaussian_var"]
    else:
        gaussian_var = None
    
    if SpeckleNoise:
        speckle_factor = config["speckle_factor"] 
    else:
        speckle_factor = None
    
    if rotation:
        rotation_degrees = config["rotation_degrees"]
    else:
        rotation_degrees = None
    
    if contrast:
        contrast_range = config["contrast_range"]
    else:
        contrast_range = (1,1)

    if config["TempDim"]:
    
        height = int(config["dataset_size"].split("x")[0])
        width = int(config["dataset_size"].split("x")[1])
        
        train_dataset = TemporalBasicDataset(dir_img, dir_mask, img_scale, mask_suffix="m_*_", rotation=rotation, vertical_flip=verticalFlip, contrast=contrast, 
                                             window_height = height, window_width = width, gaussian =GaussianNoise, speckle= SpeckleNoise, gaussian_var = gaussian_var, speckle_factor = speckle_factor, contrast_range = (contrast_range[0], contrast_range[1]), rot_degree = rotation_degrees)
        
        val_dataset = TemporalBasicDataset(dir_img_val, dir_mask_val, img_scale, mask_suffix="m_*_", 
                                             window_height = height, window_width = width)
    elif config["BGS_input"]:
        
        train_dataset = BGBasicDataset(dir_img, dir_mask, dir_bgs, img_scale, mask_suffix="crop_m_*_", rotation=rotation, vertical_flip=verticalFlip, contrast=contrast, gaussian =GaussianNoise, speckle= SpeckleNoise, gaussian_var = gaussian_var, speckle_factor = speckle_factor, contrast_range = (contrast_range[0], contrast_range[1]), rot_degree =rotation_degrees)
        
        val_dataset = BGBasicDataset(dir_img_val, dir_mask_val, dir_bgs_val, img_scale, mask_suffix="crop_m_*_")
    else:
        #, gaussian_var: float =  0.001, speckle_factor : float =  0.1, contrast_range: tuple = (1, 1.5), rot_degree: int = 10
        train_dataset = BasicDataset(dir_img, dir_mask, img_scale, mask_suffix="crop_m_*_", rotation=rotation, vertical_flip=verticalFlip, contrast=contrast, gaussian =GaussianNoise, speckle= SpeckleNoise, gaussian_var = gaussian_var, speckle_factor = speckle_factor, contrast_range = (contrast_range[0], contrast_range[1]), rot_degree = rotation_degrees )
        
        val_dataset = BasicDataset(dir_img_val, dir_mask_val, img_scale, mask_suffix="crop_m_*_")


    n_val = len(val_dataset)
    n_train = len(train_dataset)

    loader_args = dict(batch_size=config["batch_size"], num_workers=1, pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_channels = 1
    if config["TempDim"]:
        n_channels = 3
    elif config["BGS_input"]:
        n_channels = 2
    filters = config["filters"]
    model = UNet(n_channels=n_channels, n_classes=4, filters=filters, bilinear=False)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        
    model.to(device=device)

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=config["patience"], min_lr=1e-6)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=config["amp"])
    global_step = 0
    
    best_val_score = -np.inf
    best_model_state_dict = None
    train_losses = []
    train_scores = []
    val_scores = []
    
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        
        epoch_loss = 0
        epoch_train_score = 0
        train_count = 0
        val_score = 0
        
        epochs = config["epochs"]
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=config["amp"]):
                    masks_pred = model(images)

                    pred_softmax = F.softmax(masks_pred, dim=1).float()
                    target_one_hot = F.one_hot(true_masks, num_classes=4).permute(0, 3, 1, 2).float()

                    loss = jaccard2_loss(target_one_hot[:,1:], pred_softmax[:,1:])
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

        epoch_loss = epoch_loss / train_count
        
        train_losses.append(epoch_loss / train_count)
        train_scores.append(epoch_train_score / train_count)

        model.eval()
        jaccard_score = 0
        num_val_batches = len(val_loader)
        
        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=config["amp"]):
            for batch_idx, batch in enumerate(tqdm(val_loader, total=num_val_batches, desc='Validation round', unit='batch', leave=True)):
                image, mask_true = batch['image'], batch['mask']
                
                image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                mask_true = mask_true.to(device=device, dtype=torch.long)

                mask_pred = model(image)
                mask_true = F.one_hot(mask_true, 4).permute(0, 3, 1, 2).float()
                mask_pred = F.softmax(mask_pred, dim=1).float()

                jaccard_score += jaccard2_coef(mask_true[:,1:], mask_pred[:,1:]).item()

        val_score = jaccard_score / num_val_batches
        val_scores.append(val_score)

        # Update learning rate
        scheduler.step(val_score)
        
        if val_score > best_val_score:
            best_val_score = val_score
            best_model_state_dict = model.state_dict()
            trial.set_user_attr('best_model_state_dict', best_model_state_dict)
            trial.set_user_attr('best_epoch', epoch)
             

        # Memory management
        torch.cuda.empty_cache()
        del loss, masks_pred, pred_softmax, target_one_hot

        trial.report(val_score, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        
        trial.set_user_attr('train_losses', train_losses)
        trial.set_user_attr('train_scores', train_scores)
        trial.set_user_attr('val_scores', val_scores)
        trial.set_user_attr('config', config)

    return val_score
    
def get_search_space_str(config):
    config_str = "\n".join([f"{k}: {v}" for k, v in config.items()])
    return config_str

# Create and run the Optuna study
if __name__ == "__main__":

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    session_folder = f'training_session_{timestamp}'
    session_dir = Path(session_folder)
    session_dir.mkdir(parents=True, exist_ok=True)

    # Get the current script's path 
    current_script = os.path.abspath(__file__) 
    # Define the destination folder and filename 
    destination_folder = session_folder
    destination_file = os.path.join(destination_folder, os.path.basename(current_script)) 
    # Copy the script to the destination folder 
    shutil.copyfile(current_script, destination_file)

    #random_seed = 1
    #torch.backends.cudnn.enabled = False  # Disable cuDNN use of nondeterministic algorithms
    #torch.manual_seed(random_seed)
    
    #Rsampler = optuna.samplers.RandomSampler()
    sampler = optuna.samplers.TPESampler()
    
    study = optuna.create_study(direction="maximize", sampler = sampler)
    study.optimize(objective, n_trials=20)
    
    
     # Find number of pruned and completed trials
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    
    # Display the study statistics
    print("\nStudy statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    best_trial = study.best_trial
    
    print("config pqrqmeters seqrch space : ")
    config = best_trial.user_attrs['config']
    config_str = get_search_space_str(config)
    print("Search Space Config:\n", config_str)
    
    
    best_model_state_dict = best_trial.user_attrs['best_model_state_dict']
    best_epoch = best_trial.user_attrs['best_epoch']
    train_losses = best_trial.user_attrs['train_losses']
    train_scores = best_trial.user_attrs['train_scores']
    val_scores = best_trial.user_attrs['val_scores']
    
    
    
    
    file_name = 'ModelHyperparameters.txt'
    
    
    
    model_path = os.path.join(session_dir,f'best_model_epoch{best_epoch}.pth')
    torch.save(best_model_state_dict, model_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path
    print(f'Loading model')
    print(f'Using device {device}')
    filters = (64, 128, 256, 512, 1024)
    
    if best_trial.params.get('TempDim'):
        model = UNet(n_channels=3, n_classes=4, filters = filters)
    elif best_trial.params.get('BGS_input'):
        model = UNet(n_channels=2, n_classes=4, filters = filters)
    else:
        model = UNet(n_channels=1, n_classes=4, filters = filters)
    

    model.to(device=device)
    
    
    
    mask_values = best_model_state_dict.pop('mask_values', [0, 1])
    
    if any(key.startswith('module.') for key in best_model_state_dict.keys()):
        # The model was saved using DataParallel, so we wrap the model with DataParallel
        model = torch.nn.DataParallel(model)
        model.load_state_dict(best_model_state_dict)
    else:
        # The model was not saved using DataParallel, so we can load directly
        model.load_state_dict(best_model_state_dict)
    
    print(('Model loaded!'))

    test_model(model, device, folder = session_folder, TempDim = best_trial.params.get('TempDim'), BGS_input=best_trial.params.get('BGS_input'), datasize = best_trial.params.get('dataset_size'))
    
    # Plot the losses and scores
    plt.figure()
    plt.plot(np.arange(len(train_losses)), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.savefig(os.path.join(session_folder,'train_loss.png' ))
    
    plt.figure()
    plt.plot(np.arange(len(val_scores)), val_scores, label='Validation Jaccard Score')
    plt.plot(np.arange(len(train_scores)), train_scores, label='Training Jaccard Score')
    plt.xlabel('Epoch')
    plt.ylabel('Jaccard Score')
    plt.title('Jaccard Scores')
    plt.legend()
    
    plt.savefig(os.path.join(session_folder, 'jaccard_scores.png'))
    
    
    
    file_name = 'ModelHyperparameters.txt'
    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
        
        write_to_file(session_folder, file_name, f"    {key}: {value}")
        
    
     # Save results to csv file
    df = study.trials_dataframe().drop(['datetime_start', 'datetime_complete', 'user_attrs_best_epoch', 'user_attrs_best_model_state_dict', 'user_attrs_train_losses', 'user_attrs_train_scores', 'user_attrs_val_scores', 'user_attrs_config' ], axis=1)  # Exclude columns
    #df = df.loc[df['state'] == 'COMPLETE']        # Keep only results that did not prune
    #df = df.drop('state', axis=1)                 # Exclude state column
    df = df.sort_values('value')                  # Sort based on accuracy
    csv_path = session_dir / f'optuna_results_{timestamp}.csv'
    df.to_csv(csv_path, index=False)  # Save to csv file

    # Display results in a dataframe
    
    print("\nOverall Results (ordered by accuracy):\n {}".format(df))
    
    evaluator = FanovaImportanceEvaluator()
    # Find the most important hyperparameters
    most_important_parameters = optuna.importance.get_param_importances(study, target=None, evaluator = evaluator)

    # Display the most important hyperparameters
    print('\nMost important hyperparameters:')
    for key, value in most_important_parameters.items():
        print('  {}:{}{:.2f}%'.format(key, (15-len(key))*' ', value*100))
    
     
