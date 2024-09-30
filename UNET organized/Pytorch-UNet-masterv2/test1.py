import argparse
import torch
from torch.utils.data import DataLoader
from evaluate3 import evaluate
from unet import UNet
from utils.data_loading import BasicDataset
from utils.data_loading_bg import BGBasicDataset
from utils.spaciotemporal_data_loading import TemporalBasicDataset



def test_model(model, device, folder,  datasize, batch_size: int = 10,amp: bool = False, TempDim: bool = False, BGS_input: bool = False, test = False ):

    model.eval()
    
    """The reason why the loadingfolder of the images changes according to the unet method, is due to the special way the 3 frames version has to be cropped in order to centered.
    Also due to how the files were named and in the datasets created for each version. This can probably be made easier to work around. 
    It would need to verify the dataset imaages names and datasets .py"""
    if not test:
        if not TempDim:
            dir_img = f'{datasize}/data_val/imgs/'
            dir_mask = f'{datasize}/data_val/masks/'
        else: 
            dir_img = 'FullSize1276x664/data_val/imgs/'
            dir_mask = 'FullSize1276x664/data_val/masks/'
            
        if BGS_input:
            dir_bgs = f'{datasize}/data_val/bgs/'
    if test:
        if not TempDim:
            dir_img = f'{datasize}/data_test/imgs/'
            dir_mask = f'{datasize}/data_test/masks/'
        else: 
            dir_img = 'FullSize1276x664/data_test/imgs/'
            dir_mask = 'FullSize1276x664/data_test/masks/'
            
        if BGS_input:
            dir_bgs = f'{datasize}/data_test/bgs/'
    
    
    
    
    
    if TempDim:
        height = int(datasize[1:].split("x")[0])
        width = int(datasize[1:].split("x")[1])
        test_dataset = TemporalBasicDataset(dir_img, dir_mask, 1, mask_suffix="m_*_", window_height=height, window_width=width)
    elif BGS_input:
        test_dataset = BGBasicDataset(dir_img, dir_mask, dir_bgs, 1, mask_suffix="crop_m_*_")
    else:
        test_dataset = BasicDataset(dir_img, dir_mask, 1, mask_suffix="crop_m_*_")

    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    n_test = len(test_dataset)

    val_score = evaluate(net = model, dataloader= test_loader, device=device, amp= amp,session_folder=folder, division="test" )

    print("The Jaccard score for the test of the model was: ", str(val_score), ", from ",str(n_test), " test images")




def get_args():
    parser = argparse.ArgumentParser(description='test_evaluate masks from input images')
    parser.add_argument('--model', '-m', default=False, metavar='FILE',
                        help='Specify the file in which the model is stored')
    
    parser.add_argument('--mask_threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=4, help='Number of classes')
    parser.add_argument('--TempDim' , action='store_true', default=False, help='Temporal dimension: Past, current and next frame')
    parser.add_argument('--dataset_size', '-d', type=str, choices=['500x500', '750x664', '350x350', '1276x664'], default='350x350', help='Dataset size to use')
    parser.add_argument('--BGS_input' , action='store_true', default=False, help='Using BGS together with the input, 2D input')
    parser.add_argument('--test' , action='store_true', default=False, help='test or validation')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading model {args.model}')
    print(f'Using device {device}')
    filters = (64, 128, 256, 512, 1024)
    if args.TempDim:
        model = UNet(n_channels=3, n_classes=args.classes, filters = filters)
    elif args.BGS_input:
        model = UNet(n_channels=2, n_classes=args.classes, filters = filters)
    else:
        model = UNet(n_channels=1, n_classes=args.classes, filters = filters)
    

    model.to(device=device)
    
    

    state_dict = torch.load(args.model, map_location=device)
        
    if any(key.startswith('module.') for key in state_dict.keys()):
            # The model was saved using DataParallel, so we wrap the model with DataParallel
        model = torch.nn.DataParallel(model)
        
        model.load_state_dict(state_dict)
    else:
            # The model was not saved using DataParallel, so we can load directly
        model.load_state_dict(state_dict)
        
        
    print(f'Model loaded from {args.model}')
    
    test_model(model = model, device = device, folder = args.model[:32], TempDim = args.TempDim, BGS_input=args.BGS_input, datasize = args.dataset_size, test = args.test)
    
    






