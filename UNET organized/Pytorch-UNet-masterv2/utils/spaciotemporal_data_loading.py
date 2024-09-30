import logging
import numpy as np
import torch
from PIL import Image, ImageOps, ImageEnhance
from os.path import splitext
from pathlib import Path
from torch.utils.data import Dataset
import cv2
import random
import matplotlib.pyplot as plt
import re
import os
from skimage.util import random_noise
from datetime import datetime

def remove_first_matching_substring(original_string, substrings_to_remove):
    for substring in substrings_to_remove:
        if substring in original_string:
            return original_string.replace(substring, '')
    return original_string

def random_vertical_flip(images, mask):
    if random.random() < 0.5:
        images = [ImageOps.mirror(image) for image in images]
        mask = ImageOps.mirror(mask)
    return images, mask

def random_rotation(images, mask, degrees = 10):
    angle = random.uniform(-degrees, degrees)
    images = [image.rotate(angle, resample=Image.BILINEAR) for image in images]
    mask = mask.rotate(angle, resample=Image.NEAREST)
    return images, mask


def random_contrast(images, contrast_range=(1, 1.5)):
    
    if random.random() < 0.5:
        contrasted_images = []
       
        for image in images:
            # Convert PIL image to numpy array
            np_image = np.array(image)
    
            # Normalize image to [0, 1] range if needed
            if np_image.max() > 1:
                np_image = np_image / 255.0
    
            # Apply contrast enhancement
            factor = random.uniform(*contrast_range)
            image = Image.fromarray((np_image * 255).astype(np.uint8))  # Convert to [0, 255] for processing
            enhanced_image = ImageEnhance.Contrast(image).enhance(factor)
        
            # Convert back to numpy array
            np_enhanced = np.array(enhanced_image) / 255.0  # Convert back to [0, 1] range
            np_enhanced = np.clip(np_enhanced, 0, 1)  # Ensure pixel values are in [0, 1] range
    
            # Convert back to [0, 255] range and to uint8
            contrasted_image = Image.fromarray((np_enhanced * 255).astype(np.uint8))
            contrasted_images.append(contrasted_image)
    
        return contrasted_images
    else:
        return images



def add_gaussian_noise(images, var=0.001):

    if random.random() < 0.5:
        noisy_images = []
        for image in images:
            # Convert PIL image to numpy array
            np_image = np.array(image)
    
            # Normalize image to [0, 1] range if needed
            if np_image.max() > 1:  # Assuming 8-bit image if max pixel value > 1
                np_image = np_image / 255.0  # Convert to [0, 1] range
    
            # Add Gaussian noise
            noisy_image = random_noise(np_image, mode='gaussian', var=var)
    
            # Clip pixel values to ensure they remain in [0, 1]
            noisy_image = np.clip(noisy_image, 0, 1)
    
            # Convert back to [0, 255] range if needed
            if image.mode == 'L':  # Assuming original was 8-bit grayscale
                noisy_image = (noisy_image * 255).astype(np.uint8)
    
            # Convert numpy array back to PIL image
            noisy_pil_image = Image.fromarray(noisy_image)
            noisy_images.append(noisy_pil_image)
    
        return noisy_images
    else:
        return images



def add_speckle_noise(images, noise_factor=0.1):
    # Convert PIL image to numpy array
    if random.random() < 0.5:
        noisy_images = []
    
        for image in images:
            np_image = np.array(image)
    
            # Normalize image to [0, 1] range if needed
            if np_image.max() > 1:
                np_image = np_image / 255.0  # Convert to [0, 1] range
    
            # Generate speckle noise
            noise = np.random.normal(0, noise_factor, np_image.shape)
    
            # Add speckle noise to the image
            noisy_image = np_image + np_image * noise
    
            # Clip the values to [0, 1] range
            noisy_image = np.clip(noisy_image, 0, 1)
    
            # Convert back to [0, 255] range and appropriate data type if needed
            if image.mode == 'L':  # Assuming the image was 8-bit grayscale
                noisy_image = (noisy_image * 255).astype(np.uint8)
    
            # Convert numpy array back to PIL image
            noisy_pil_image = Image.fromarray(noisy_image)
            noisy_images.append(noisy_pil_image)
    
        return noisy_images
    else:
        return images





def load_image(filename):
    ext = splitext(str(filename))[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        filenamepath = str(filename)
        if "m_" in filenamepath:
            mask = cv2.imread(filenamepath)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY).astype(np.uint8)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            if "m_Trash" in filenamepath:
                mask = np.where(mask == 255, 1, 0)
            elif "m_SmallFish" in filenamepath:
                mask = np.where(mask == 255, 2, 0)
            elif "m_Eel" in filenamepath:
                mask = np.where(mask == 255, 3, 0)
            mask = mask.astype(np.uint8)
            mask = Image.fromarray(mask)
            return mask
        elif "_BGS_" in filenamepath:
            mask = cv2.imread(filenamepath)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY).astype(np.uint8)
            _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

            mask = mask.astype(np.uint8)
            mask = Image.fromarray(mask)
            return mask
        else:
            image = Image.open(filenamepath).convert("L")
            return image

def extract_frame_info(filename):
    # Updates the pattern to handle files that start with 'm_' followed by the information
    # This is due to the way the images were named with the special arcing cases, specifically for the 3 frame mode since you have to load images in order.
    pattern = r'm_(Trash(?:_Arcing)?|SmallFish(?:_Arcing)?|Eel(?:_Arcing)?)_(\d{4}-\d{2}-\d{2})_(\d{6})_t(\d+)_Obj_frame(\d+).jpg'
    match = re.search(pattern, filename)
    if match:
        category, date_str, time_str, track_str, frame_str = match.groups()
        category = category.split('_')[0]
        datetime_str = f'{date_str}_{time_str}'
        datetime_obj = datetime.strptime(datetime_str, '%Y-%m-%d_%H%M%S')
        frame_num = int(frame_str)
        track_num = int(track_str)
        return datetime_obj, track_num, frame_num
    return None, None, None


def sort_key(filename):
    datetime_obj, track_num, frame_num = extract_frame_info(filename)
    return (datetime_obj, track_num, frame_num)

def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(mask_suffix + str(idx)[:] + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')

# get_bounding_box and crop were defined for the this specific cqse of the 3 frames ,to be able to train cropped version in the 3 frames version. 
# The crop coordinates has to be always in reference to the current frame for the previous and next frame.
def get_bounding_box(mask):
    # Ensure the mask is a NumPy array
    mask = np.asarray(mask)
   
    # Assertions to validate the input mask
    assert mask.ndim == 2, f'Expected a 2D mask, but got a {mask.ndim}D array.'
    assert mask.dtype == bool or mask.dtype == np.bool_ or mask.dtype == np.uint8 or mask.dtype == int, f'Expected mask of type bool, np.bool_, uint8, or int, but got {mask.dtype}.'

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
   
    # Assertions to check that the mask is not empty
    assert np.any(rows), 'The mask contains no foreground pixels (rows).'
    assert np.any(cols), 'The mask contains no foreground pixels (cols).'

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Assertions to check the bounding box coordinates
    assert rmin <= rmax, f'rmin should be less than or equal to rmax, but got rmin={rmin}, rmax={rmax}.'
    assert cmin <= cmax, f'cmin should be less than or equal to cmax, but got cmin={cmin}, cmax={cmax}.'

    return rmin, rmax, cmin, cmax

def crop(image, rmin, rmax, cmin, cmax, window_height, window_width):
    obj_height = rmax - rmin + 1
    obj_width = cmax - cmin + 1

    center_r = (rmin + rmax) // 2
    center_c = (cmin + cmax) // 2

    half_window_height = window_height // 2
    half_window_width = window_width // 2

    start_r = max(center_r - half_window_height, 0)
    start_c = max(center_c - half_window_width, 0)
    end_r = min(center_r + half_window_height, image.size[1])
    end_c = min(center_c + half_window_width, image.size[0])

    if start_r == 0:
        end_r = min(window_height, image.size[1])
    if start_c == 0:
        end_c = min(window_width, image.size[0])
    if end_r == image.size[1]:
        start_r = max(0, image.size[1] - window_height)
    if end_c == image.size[0]:
        start_c = max(0, image.size[0] - window_width)

    cropped = image.crop((start_c, start_r, end_c, end_r))

    return cropped



class TemporalBasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = '', rotation: bool = False, 
                 vertical_flip: bool = False, contrast: bool = False, gaussian: bool = False, speckle: bool = False, 
                 window_height: int = 350, window_width: int = 350, gaussian_var: float =  0.006, speckle_factor : float =  0.4, 
                 contrast_range: tuple = (0.6, 1.5), rot_degree: int = 15):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.rotation = rotation
        self.vertical_flip = vertical_flip
        self.contrast = contrast
        self.gaussian = gaussian
        self.speckle = speckle
        self.window_height = window_height
        self.window_width = window_width
       
        self.gaussian_var = gaussian_var
        self.speckle_factor = speckle_factor
        self.contrast_range = contrast_range
        self.rot_degree = rot_degree

        self.mask_files = list(self.mask_dir.glob(mask_suffix + '*'))
        if not self.mask_files:
            raise RuntimeError(f'No mask files found in {mask_dir}, make sure you put your masks there')

        self.mask_files.sort(key=lambda f: extract_frame_info(f.name))
        logging.info(f'Creating dataset with {len(self.mask_files)} masks')

        self.track_frame_indices = self._group_frames_by_track()

    def _group_frames_by_track(self):
        track_frame_indices = {}
        for i, mask_file in enumerate(self.mask_files):
            datetime_obj, track_num, frame_num = extract_frame_info(mask_file.name)
            if datetime_obj is not None:
                key = (datetime_obj, track_num)
                if key not in track_frame_indices:
                    track_frame_indices[key] = []
                track_frame_indices[key].append(i)
        return track_frame_indices

    def __len__(self):
        return sum(len(indices) - 2 for indices in self.track_frame_indices.values())

    @staticmethod
    def preprocess(np_img, scale, is_mask):
        w, h = np_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        np_img = np_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(np_img)

        if is_mask:
            return img

        if not is_mask:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        for key, indices in self.track_frame_indices.items():
            if idx < len(indices) - 2:
                curr_idx = indices[idx + 1]
                prev_idx = curr_idx - 1
                next_idx = curr_idx + 1
                break
            idx -= (len(indices) - 2)
        else:
            raise IndexError(f"Index {idx} out of range for the dataset")

        curr_mask_file = self.mask_files[curr_idx]
        prev_mask_file = self.mask_files[prev_idx]
        next_mask_file = self.mask_files[next_idx]

        substrings_to_remove = ['m_SmallFish_Arcing_', 'm_SmallFish_', 'm_Eel_Arcing_', 'm_Eel_', 'm_Trash_']

        prev_img_file = list(self.images_dir.glob(remove_first_matching_substring(prev_mask_file.stem, substrings_to_remove) + '.*'))
        curr_img_file = list(self.images_dir.glob(remove_first_matching_substring(curr_mask_file.stem, substrings_to_remove) + '.*'))
        next_img_file = list(self.images_dir.glob(remove_first_matching_substring(next_mask_file.stem, substrings_to_remove) + '.*'))
        mask_file = list(self.mask_dir.glob(curr_mask_file.stem[:] + '.*'))

        assert len(prev_img_file) == 1, f'Either no image or multiple images found for the ID {prev_mask_file.stem}: {prev_img_file}'
        assert len(curr_img_file) == 1, f'Either no image or multiple images found for the ID {curr_mask_file.stem}: {curr_img_file}'
        assert len(next_img_file) == 1, f'Either no image or multiple images found for the ID {next_mask_file.stem}: {next_img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {curr_mask_file.stem}: {mask_file}'

        prev_img = load_image(prev_img_file[0])
        curr_img = load_image(curr_img_file[0])
        next_img = load_image(next_img_file[0])
        mask = load_image(mask_file[0])

        rmin, rmax, cmin, cmax = get_bounding_box(mask)
       
        # Apply the cropping to the images and mask
        prev_img = crop(prev_img, rmin, rmax, cmin, cmax, self.window_height, self.window_width)
        curr_img = crop(curr_img, rmin, rmax, cmin, cmax, self.window_height, self.window_width)
        next_img = crop(next_img, rmin, rmax, cmin, cmax, self.window_height, self.window_width)
        mask = crop(mask, rmin, rmax, cmin, cmax, self.window_height, self.window_width)

        images = [prev_img, curr_img, next_img]

        if self.vertical_flip:
            images, mask = random_vertical_flip(images, mask)
        if self.rotation:
            images, mask = random_rotation(images, mask, degrees = self.rot_degree)
        if self.contrast:
            images = random_contrast(images, contrast_range = self.contrast_range)
        if self.gaussian:
            images = add_gaussian_noise(images, var = self.gaussian_var)
        if self.speckle:
            images = add_speckle_noise(images, noise_factor = self.speckle_factor)


        images = [self.preprocess(img, self.scale, is_mask=False) for img in images]
        mask = self.preprocess(mask, self.scale, is_mask=True)

        stacked_img = np.concatenate(images, axis=0)
        return {
            'image': torch.as_tensor(stacked_img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'file_paths': {
                'prev_img_file': str(prev_img_file[0]),
                'curr_img_file': str(curr_img_file[0]),
                'next_img_file': str(next_img_file[0]),
                'mask_file': str(mask_file[0])
            }
        }

    def visualize_sample(self, idx): # This was added so i could directly watch if the images were being loaded correctly, as well as if the transformation were done correctly
        sample = self.__getitem__(idx)
        images = sample['image']
        mask = sample['mask']
        file_paths = sample['file_paths']
        pattern = r'frame(\d+)'
        frame_numbers = {}

        for key in ['prev_img_file', 'curr_img_file', 'next_img_file']:
            match = re.search(pattern, os.path.basename(file_paths[key]))
            if match:
                frame_number = match.group(1)
                frame_numbers[key] = frame_number
                print(f"{key.replace('_file', '').replace('_', ' ').title()} frame number: {frame_number}")
            else:
                print(f"{key.replace('_file', '').replace('_', ' ').title()} frame number not found")

        prev_img_number = frame_numbers.get('prev_img_file')
        curr_img_number = frame_numbers.get('curr_img_file')
        next_img_number = frame_numbers.get('next_img_file')

        fig, axs = plt.subplots(1, 5, figsize=(20, 5))
        for i in range(3):
            axs[i].imshow(images[i], cmap='gray')
            axs[i].set_title([str(prev_img_number), str(curr_img_number), str(next_img_number)][i])
            axs[i].axis('off')

        axs[3].imshow(np.transpose(images, (1, 2, 0)))
        axs[3].set_title('Together')
        axs[3].axis('off')

        axs[4].imshow(mask, cmap='gray')
        axs[4].set_title('Mask')
        axs[4].axis('off')

        plt.show()

        print(f"Previous image file: {file_paths['prev_img_file']}")
        print(f"Current image file: {file_paths['curr_img_file']}")
        print(f"Next image file: {file_paths['next_img_file']}")
        print(f"Mask file: {file_paths['mask_file']}")