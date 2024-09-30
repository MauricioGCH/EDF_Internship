import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from evaluate3 import Visualize_Preds
from unet import UNet
import torch
from utils.data_loading import BasicDataset
from utils.data_loading_bg import BGBasicDataset
from utils.spaciotemporal_data_loading import TemporalBasicDataset


# A COPY of test_model function modified to just load the dataset and create the visualization of the predictions.

def test_model(model, device, folder, batch_size: int = 3,amp: bool = False, TempDim: bool = False, BGS_input: bool = False, dataset_size=[350,350], ModLoss = False):

    model.eval()
    # ACA ESTAR PENDIENTE QUE toca CAMBIar ENTRE VAL Y TEST manualmente
    dir_img = Path(f'./{dataset_size}/data_val/imgs/')
    dir_mask = Path(f'./{dataset_size}/data_val/masks/')
    if BGS_input:
        dir_bgs = Path(f'./{dataset_size}/data_val/bgs/')
    
    if TempDim:
        dir_img = Path('FullSize1276x664/data_val/imgs/')
        dir_mask = Path('FullSize1276x664/data_val/masks/')
        height = int(dataset_size.split("x")[0])
        width = int(dataset_size.split("x")[1])
        test_dataset = TemporalBasicDataset(dir_img, dir_mask, 1, mask_suffix="m_*_", window_height = height, window_width = width)
    elif BGS_input:
        test_dataset = BGBasicDataset(dir_img, dir_mask, dir_bgs, 1, mask_suffix="crop_m_*_")
    else:
        test_dataset = BasicDataset(dir_img, dir_mask, 1, mask_suffix="crop_m_*_") 

    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=True, **loader_args)
    
    Visualize_Preds(net= model, dataloader= test_loader, device=device, amp = amp, epoch = 0,session_folder=folder, division="test" ) # Set to "division" set test to also obtain the confusion matrix and Classification report


def get_args():
    parser = argparse.ArgumentParser(description='test_evaluate masks from input images')
    parser.add_argument('--model', '-m', default=False, metavar='FILE',
                        help='Specify the file in which the model is stored')

    parser.add_argument('--scale', '-s', type=float, default=1,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=4, help='Number of classes')
    parser.add_argument('--TempDim' , action='store_true', default=False, help='Temporal dimension: Past, current and next frame')
    parser.add_argument('--dataset_size', '-d', type=str, choices=['500x500', '750x664', '350x350', '1276x664'], default='350x350', help='Dataset size to use')
    parser.add_argument('--BGS_input' , action='store_true', default=False, help='Using BGS together with the input, 2D input')

    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading model {args.model}')
    print(f'Using device {device}')
    if args.TempDim:
        model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    elif args.BGS_input:
        model = UNet(n_channels=2, n_classes=args.classes, bilinear=args.bilinear)
    else:
        model = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)

    model.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    #mask_values = state_dict.pop('mask_values', [0, 1])
    
    # as some models were training using parllel training but other didn't, this is to be able to load them without problem
    if any(key.startswith('module.') for key in state_dict.keys()):
        # The model was saved using DataParallel, so we wrap the model with DataParallel
        model = torch.nn.DataParallel(model)
        # Remove the 'module.' prefix
        
        model.load_state_dict(state_dict)
    else:
        # The model was not saved using DataParallel, so we can load directly
        model.load_state_dict(state_dict)
    
    #model.load_state_dict(state_dict)
    print(('Model loaded!'))
    print(args.model[:32])
    print(args.model.split("/")[0])
    
    # folder = args.model[:32] is set to 32 due to how the model paths are expected to be loaded, the 32 characters correspond to the folder name
    test_model(model, device, folder = args.model[:32], TempDim = args.TempDim, BGS_input=args.BGS_input, dataset_size= args.dataset_size)