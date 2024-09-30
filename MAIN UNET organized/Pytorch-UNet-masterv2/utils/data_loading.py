import logging
import numpy as np
import torch
from PIL import Image, ImageOps, ImageEnhance
from functools import partial
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import cv2
import random
import matplotlib.pyplot as plt
from skimage.util import random_noise




def random_vertical_flip(image, mask):
     
    # Randomly decide whether to flip vertically
    if random.random() < 0.5:
        image = ImageOps.mirror(image)  # Flip image vertically
        mask = ImageOps.mirror(mask)    # Flip mask vertically
    return image, mask
   

def random_rotation(image, mask, degrees = 10):

    angle = random.uniform(-degrees, degrees)
    image = image.rotate(angle, resample=Image.BILINEAR)
    mask = mask.rotate(angle, resample=Image.NEAREST)
    return image, mask

   
def random_contrast(images, contrast_range=(1, 1.5)):
    # Convert PIL image to numpy array
    np_image = np.array(images)

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

    return contrasted_image
   

def add_gaussian_noise(image, var=0.001):
    # Convert PIL image to numpy array
    
    if random.random() < 0.5:
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

        return noisy_pil_image
    else:
        return image

def add_speckle_noise(images, noise_factor=0.1):
    # Convert PIL image to numpy array
    
    if random.random() < 0.5:
        np_image = np.array(images)
    
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
        if images.mode == 'L':  # Assuming the image was 8-bit grayscale
            noisy_image = (noisy_image * 255).astype(np.uint8)
    
        # Convert numpy array back to PIL image
        noisy_pil_image = Image.fromarray(noisy_image)
    
        return noisy_pil_image
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
        if "_m_" in filenamepath: # its a mask
            mask = cv2.imread(filenamepath)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY).astype(np.uint8)
            _,mask = cv2.threshold(mask,127,255, cv2.THRESH_BINARY)
           
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


def unique_mask_values(idx, mask_dir, mask_suffix):
   
   
    mask_file = list(mask_dir.glob(mask_suffix + str(idx)[5:]  + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = '', rotation: bool = False, 
                 vertical_flip: bool = False, contrast: bool = False, gaussian: bool = False, speckle: bool = False, gaussian_var: float =  0.001, 
                 speckle_factor : float =  0.1, contrast_range: tuple = (1, 1.5), rot_degree: int = 10):
        
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
        self.gaussian_var = gaussian_var
        self.speckle_factor = speckle_factor
        self.contrast_range = contrast_range
        self.rot_degree = rot_degree
       
        
        # List all files in the directory
        files = [file for file in listdir(images_dir) 
                 if isfile(join(images_dir, file)) and not file.startswith('.')]
        
        # Sort files alphabetically
        files.sort()
        
        # Extract file names without extensions
        self.ids = [splitext(file)[0] for file in files]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
       

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(np_img, scale, is_mask):
         
        w, h = np_img.size

        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        np_img = np_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(np_img)

        if is_mask:
            mask = img
            # mask = np.zeros((newH, newW), dtype=np.int64)
            # for i, v in enumerate(mask_values):
            #     if img.ndim == 2:
            #         mask[img == v] = i
            #     else:
            #         mask[(img == v).all(-1)] = i

            return mask

        if not is_mask:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        NoCropName = name[5:]
        mask_file = list(self.mask_dir.glob(self.mask_suffix + NoCropName  + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
       
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])
       

        if self.vertical_flip:
            img, mask = random_vertical_flip(img, mask)
        if self.rotation:
            img, mask = random_rotation(img, mask, degrees = self.rot_degree)
        if self.contrast:
            img= random_contrast(img, contrast_range = self.contrast_range )
        if self.gaussian:
            img = add_gaussian_noise(img, var = self.gaussian_var)
        if self.speckle:
            img = add_speckle_noise(img, noise_factor = self.speckle_factor)
        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)


        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'filename': str(img_file[0])
        }
    
    def visualize_sample(self, idx): # This was added so i could diretcly watch if the images were being loaded corretcly, as well as if the transformation were done correctly
        sample = self.__getitem__(idx)
        combined_image = sample['image'][0].numpy()
        mask = sample['mask'].numpy()
        
       

        fig, axs = plt.subplots(1, 2, figsize=(20, 5))
       
        axs[0].imshow(combined_image, cmap='gray')
        axs[0].set_title('Original Image')
        axs[0].axis('off')

        axs[1].imshow(mask, cmap='gray')
        axs[1].set_title('Mask')
        axs[1].axis('off')

        plt.show()