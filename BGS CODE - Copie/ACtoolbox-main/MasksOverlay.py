import cv2
import numpy as np
import glob
import os
from src.utils import random_color

def draw_mask_edges(image, mask, color=(0, 255, 0), thickness=2):
    """
    Dibuja los contornos de una máscara sobre una imagen con un color específico y grosor.

    :param image: Imagen original.
    :param mask: Máscara binaria (0 y 255).
    :param color: Color de los contornos en formato (B, G, R).
    :param thickness: Grosor de los contornos.
    :return: Imagen con los contornos de la máscara dibujados.
    """
    # Encuentra los contornos de la máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Dibuja los contornos sobre la imagen original
    cv2.drawContours(image, contours, -1, color, thickness)
    
    return image

def overlay_contours(image, masks, colors, thickness=2):
    """
    Superpone los contornos de varias máscaras sobre una imagen con diferentes colores y grosor.

    :param image: Imagen original.
    :param masks: Lista de máscaras binarias (0 y 255).
    :param colors: Lista de colores para los contornos de cada máscara en formato (B, G, R).
    :param thickness: Grosor de los contornos.
    :return: Imagen con todos los contornos de las máscaras dibujados.
    """
    output_image = image.copy()
    
    for mask, color in zip(masks, colors):
        output_image = draw_mask_edges(output_image, mask, color, thickness)
    
    return output_image

# Ejemplo de uso
if __name__ == "__main__":
    # Leer la imagen original
    original_image = cv2.imread(r"C:\Users\chapi\Documents\STAGE\CODE\bgslibrary\ACtoolbox-main\Dataset\Small_ARIS_Mauzac\TEST\All_Originals\2014-11-05_184000_t4_Obj_frame1323.jpg")

    # Crear algunas máscaras de ejemplo
  

    # Lista de máscaras
    #"C:\Users\chapi\Documents\STAGE\CODE\bgslibrary\ACtoolbox-main\Dataset\Small_ARIS_Mauzac\TEST\All_Originals\NewMasks\2014-11-05_184000_t4_Obj_frame1323\m_SmallFish5_2014-11-05_184000_t4_Obj_frame1323.jpg"
    masks_paths= glob.glob(r"C:\Users\chapi\Documents\STAGE\CODE\bgslibrary\ACtoolbox-main\Dataset\Small_ARIS_Mauzac\TEST\All_Originals\NewMasks\2014-11-05_184000_t4_Obj_frame1323\m_SmallFish*_2014-11-05_184000_t4_Obj_frame1323.jpg")
    masks = []
    for mask_path in masks_paths:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        masks.append(mask)

    # Lista de colores (B, G, R)
    colors = []
    for i in range(len(masks_paths)):

        colors.append(random_color())

    result_image = overlay_contours(original_image, masks, colors, thickness=1)

    # Mostrar el resultado
    cv2.imshow('Result Image with Contours', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
