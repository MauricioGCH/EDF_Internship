# -*- coding: utf-8 -*-
import torch
import numpy as np
import os
import torch.nn.functional as F
import torch.utils.data
from utils.data_loading import BasicDataset
from utils.data_loading_bg import BGBasicDataset
from utils.spaciotemporal_data_loading import TemporalBasicDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dice_score import jaccard2_coef
from skimage.morphology import label
from skimage.measure import regionprops
import argparse
import logging
import cv2
from unet import UNet

def count_objects_per_class(mask):
    mask_onehot = F.one_hot(mask, num_classes=4).permute(0, 3, 1, 2).float()
    trash_channel = mask_onehot[0, 1, :, :].cpu().numpy()
    smallfish_channel = mask_onehot[0, 2, :, :].cpu().numpy()
    eel_channel = mask_onehot[0, 3, :, :].cpu().numpy()
    
    trash_count = len(regionprops(label(trash_channel)))
    smallfish_count = len(regionprops(label(smallfish_channel)))
    eel_count = len(regionprops(label(eel_channel)))
    
    return trash_count, smallfish_count, eel_count


def trackPrediction(model, BGS_input, TempDim, dataset_size, division, video, track):

    if not TempDim:
        dir_img = os.path.join(dataset_size, division, video, track, "imgs")
        dir_mask = os.path.join(dataset_size, division, video, track, "masks")
    else:
        dir_img = os.path.join("T1276x664", division, video, track, "imgs")
        dir_mask = os.path.join("T1276x664", division, video, track, "masks")
       
    if BGS_input:
        dir_bgs = os.path.join(dataset_size, division, video, track, "bgs")
    

    if TempDim:
        height = int(dataset_size[1:].split("x")[0])
        width = int(dataset_size[1:].split("x")[1])
        dataset = TemporalBasicDataset(dir_img, dir_mask, 1, mask_suffix="m_*_", window_height=height, window_width=width)
    elif BGS_input:
        dataset = BGBasicDataset(dir_img, dir_mask, dir_bgs, 1, mask_suffix="crop_m_*_")
    else:
        dataset = BasicDataset(dir_img, dir_mask, 1, mask_suffix="crop_m_*_")

    loader_args = dict(batch_size=1, num_workers=1, pin_memory=True)
    loader = DataLoader(dataset, shuffle=False, drop_last=False, **loader_args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    track_result = {}
    
    for batch_idx, batch in enumerate(tqdm(loader, total=len(loader), desc="Image track prediction", unit="batch", leave=True)):

        
        if TempDim:
            image, mask_true, filename = batch["image"], batch["mask"], batch["file_paths"]["curr_img_file"]
        elif BGS_input:
            image, mask_true, filename = batch["image"], batch["mask"], batch["file_paths"]["img_file"]
        else:
            image, mask_true, filename = batch["image"], batch["mask"], batch["filename"]
            
        image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        mask_true = mask_true.to(device=device, dtype=torch.long)

        mask_pred = model(image)
        mask_true_hot = F.one_hot(mask_true, 4).permute(0, 3, 1, 2).float()
        mask_pred = F.softmax(mask_pred, dim=1).float()

        jaccard_score = jaccard2_coef(mask_true_hot[:, 1:], mask_pred[:, 1:]).item()

        trash_pred, smallfish_pred, eel_pred = count_objects_per_class(torch.argmax(mask_pred, dim=1))
        trash_mask, smallfish_mask, eel_mask = count_objects_per_class(mask_true)

        track_result[batch_idx] = {
            "image": F.one_hot(torch.argmax(mask_pred, dim=1), 4).permute(0, 3, 1, 2).float().clone().detach().cpu(),
            "Jaccard": jaccard_score,
            "Obj Count": [[trash_pred, smallfish_pred, eel_pred], [trash_mask, smallfish_mask, eel_mask]],
            "mask": mask_true_hot.clone().detach().cpu(),
            "filename": filename,
            "input": image.clone().detach().cpu()
            
        }

    return track_result


def update_predictions(track_result):
    total_counts = np.array([0, 0, 0])

    for data in track_result.values():
        pred_counts = data["Obj Count"][0]
        total_counts += pred_counts

    most_consistent_class = np.argmax(total_counts) + 1

    for data in track_result.values():
        mask_pred_argmax = torch.argmax(data["image"].clone(), dim=1)
        new_pred = torch.where(mask_pred_argmax != 0, most_consistent_class, 0)
        #new_pred = (mask_pred_argmax == most_consistent_class).float()
        data["image"] = F.one_hot(new_pred, 4).permute(0, 3, 1, 2).float()

    return track_result


def recalculate_metrics(track_result):
    for data in track_result.values():
        

        data["Jaccard New"] = jaccard2_coef(data["image"][:, 1:], data["mask"][:, 1:]).item()
        
        pred_classes = torch.argmax(data["image"], dim=1)
        true_classes = torch.argmax(data["mask"], dim=1)
        #print(data["image"].size())
        #print(data["mask"].size())
        data["Obj Count New"] = [
            count_objects_per_class(pred_classes),
            count_objects_per_class(true_classes)
        ]
    return track_result


def generate_report(track_result, folder, filename="report.txt"):
    output_path = os.path.join(folder, filename)
   
    with open(output_path, 'w', encoding='utf-8') as file:
        old_jaccard_scores = []
        new_jaccard_scores = []
        old_total_counts = np.array([0, 0, 0])
        gt_total_counts = np.array([0, 0, 0])
        new_total_counts = np.array([0, 0, 0])

        for frame_idx, data in track_result.items():
            old_jaccard_scores.append(data["Jaccard"])
            new_jaccard_scores.append(data.get("Jaccard New", data["Jaccard"]))

            old_total_counts += np.array(data["Obj Count"][0])
            gt_total_counts += np.array(data["Obj Count"][1])
            new_total_counts += np.array(data.get("Obj Count New", data["Obj Count"])[0])
            filename = data["filename"]
            
            file.write(f"Frame {frame_idx}:\n")
            file.write(f"{filename} \n")
            file.write(f"Old Jaccard: {data['Jaccard']}, New Jaccard: {data.get('Jaccard New', data['Jaccard'])}\n")
            file.write(f"Old Obj Count: {data['Obj Count'][0]}, New Obj Count: {data.get('Obj Count New', data['Obj Count'])[0]}\n\n")
       
        avg_old_jaccard = np.mean(old_jaccard_scores)
        avg_new_jaccard = np.mean(new_jaccard_scores)

        file.write(f"Average Old Jaccard: {avg_old_jaccard}\n")
        file.write(f"Average New Jaccard: {avg_new_jaccard}\n\n")
        file.write(f"Total Object Count Before Modifications: {old_total_counts.tolist()}\n")
        file.write(f"Total Object Count After Modifications: {new_total_counts.tolist()}\n")
        file.write(f"Ground Truth Total Object Count : {gt_total_counts.tolist()}\n")


def create_video(track_result, folder, filename="output_video.mp4", fps=7):
    output_path = os.path.join(folder, filename)

    frame_height, frame_width = track_result[0]['image'].shape[2:]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width*2, frame_height))

    for data in track_result.values():
        mask_pred = (data['image'][0].numpy() * 255).astype(np.uint8)
        #print(data['input'].size())
       
        input_image = (data['input'][0, 0].numpy() * 255).astype(np.uint8)
        #print(np.max(input_image))
        #print(np.shape(input_image))

        frame = np.zeros((frame_height, frame_width*2, 3), dtype=np.uint8)
        frame[:,0:frame_width, 0] = mask_pred[3]
        frame[:,0:frame_width, 1] = mask_pred[2]
        frame[:,0:frame_width, 2] = mask_pred[1]
        
        rgb_input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
        #print(np.max(rgb_input_image))
        
        frame[:, frame_width:, :] = rgb_input_image
        

        out.write(frame)

    out.release()


def post_process(track_result, folder, output_txt="results.txt", video_before="before.mp4", video_after="after.mp4"):
    create_video(track_result, folder, video_before)
    track_result = update_predictions(track_result)
    track_result = recalculate_metrics(track_result)
    generate_report(track_result, folder, output_txt)
    create_video(track_result, folder, video_after)
    print(f"Report generated at {output_txt}, videos saved as {video_before} and {video_after}")


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--load', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--division', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--dataset_size', type=str, default=False, help='dataset folder')
    parser.add_argument('--video', type=str, default=False, help='dataset folder')
    parser.add_argument('--track', type=str, default=False, help='dataset folder')
    parser.add_argument('--output_txt', type=str, default="results.txt", help='output folder txt file')
    parser.add_argument('--video_before', type=str, default="before.mp4", help='output folder txt file')
    parser.add_argument('--video_after', type=str, default="after.mp4", help='output folder txt file')
    
    parser.add_argument('--BGS_input', action='store_true', default=False, help='Use BGS')
    parser.add_argument('--TempDim', action='store_true', default=False, help='Use SpatioTemporal')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')
    
    
            
    
    filters = (64, 128, 256, 512, 1024)
    if args.TempDim:
        model = UNet(n_channels=3, n_classes=4, filters = filters)

    elif args.BGS_input:
        model = UNet(n_channels=2, n_classes=4, filters = filters)

    else:
        model = UNet(n_channels=1, n_classes=4, filters = filters)
        
    #model = model.to(memory_format=torch.channels_last)
    model.to(device=device)
    
    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        
        if any(key.startswith('module.') for key in state_dict.keys()):
            # The model was saved using DataParallel, so we wrap the model with DataParallel
            model = torch.nn.DataParallel(model)
            
            model.load_state_dict(state_dict)
        else:
            # The model was not saved using DataParallel, so we can load directly
            model.load_state_dict(state_dict)
        
        print(f'Model loaded from {args.load}')
        
        
    
    videos = os.listdir(os.path.join(args.dataset_size, args.division))
    
    for video in videos:
            tracks = os.listdir(os.path.join(args.dataset_size, args.division, video))
            
            for track in tracks:

                track_result = trackPrediction(
                    model=model,
                    BGS_input=args.BGS_input,
                    TempDim=args.TempDim,
                    dataset_size=args.dataset_size,
                    division=args.division,
                    video=video,
                    track=track)
    
                os.makedirs(os.path.join(args.load.split("/")[0], "Track_analysis",args.dataset_size, args.division, video, track), exist_ok = True)
                folder = os.path.join(args.load.split("/")[0], "Track_analysis",args.dataset_size, args.division, video, track)
                post_process(
                    track_result,
                    folder=folder,
                    output_txt=args.output_txt,
                    video_before=args.video_before,
                    video_after=args.video_after
                )