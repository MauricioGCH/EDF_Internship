import numpy as np
from skimage import measure, morphology
import cv2
import matplotlib.pyplot as plt
import glob
import os

# Supongamos que `mask` es tu máscara de segmentación binaria
mask = cv2.imread(r"ACtoolbox-main\Binary_Obj_frame2547.png", cv2.IMREAD_GRAYSCALE)
# Etiquetar las regiones conectadas en la máscara
def filter_small_regions_by_size(image, threshold_percentage=0.25, binary_threshold=128):
    """
    Filtra las regiones pequeñas en una imagen binaria basada en el tamaño de la región más grande.

    Parámetros:
    - image_path: Ruta a la imagen.
    - threshold_percentage: El porcentaje del área de la región más grande que se usará como umbral para filtrar.
    - binary_threshold: El umbral para binarizar la imagen.

    Retorna:
    - filtered_mask: La máscara filtrada donde solo permanecen las regiones suficientemente grandes.
    """
    # Cargar la imagen en escala de grises
    #gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Asegúrate de que la imagen se ha cargado correctamente
    if image is None:
        raise ValueError("No se pudo cargar la imagen. Verifica la ruta y el formato de la imagen.")

    # Binarizar la imagen usando un umbral adecuado
    #_, mask = cv2.threshold(image, binary_threshold, 255, cv2.THRESH_BINARY)

    # Etiquetar las regiones conectadas en la máscara
    labeled_mask, num_features = measure.label(image, return_num=True)

    # Obtener las propiedades de cada región
    properties = measure.regionprops(labeled_mask)

    # Encontrar el área de la región más grande
    max_area = max([prop.area for prop in properties])

    # Calcular el umbral de tamaño
    size_threshold = max_area * threshold_percentage

    # Crear una máscara para mantener las regiones grandes
    filtered_mask = np.zeros_like(image, dtype=bool)

    # Añadir a la máscara las regiones que son mayores al umbral
    for prop in properties:
        if prop.area >= size_threshold:
            filtered_mask[labeled_mask == prop.label] = True

    # Convertir la máscara filtrada en una máscara binaria
    filtered_mask = (filtered_mask * 255).astype(np.uint8)

    return filtered_mask


paths = glob.glob(r"C:\Users\chapi\Documents\STAGE\CODE\bgslibrary\Test\ARIS_Mauzac_OtherFish\SuBSENSE\VideoModeResults\2014-11-05_184000\Track_final_Images\Foreground\track23\Binary_Obj_frame*.png")

for image in paths:
    basename = os.path.basename(image)
    filtered = filter_small_regions_by_size(image)

    cv2.imwrite(os.path.join(os.path.dirname(image),str("F"+os.path.basename(image))), filtered)

# if mask is None:
#     raise ValueError("No se pudo cargar la imagen. Verifica la ruta y el formato de la imagen.")

# # Binarizar la imagen usando un umbral adecuado
# _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

# # Etiquetar las regiones conectadas en la máscara
# labeled_mask, num_features = measure.label(mask, return_num=True)

# # Obtener las propiedades de cada región
# properties = measure.regionprops(labeled_mask)

# # Ordenar las regiones por área en orden descendente
# properties = sorted(properties, key=lambda prop: prop.area, reverse=True)

# # Número de regiones a mantener
# num_regions_to_keep = 4

# # Crear una máscara para mantener solo las x regiones más grandes
# filtered_mask = np.zeros_like(mask, dtype=bool)

# # Añadir a la máscara las x regiones más grandes
# for prop in properties[:num_regions_to_keep]:
#     filtered_mask[labeled_mask == prop.label] = True

# # Convertir la máscara filtrada en una máscara binaria
# filtered_mask = (filtered_mask * 255).astype(np.uint8)

# # Mostrar la máscara resultante
# plt.imshow(filtered_mask, cmap='gray')
# plt.title('Máscara con las x regiones más grandes')
# plt.show()