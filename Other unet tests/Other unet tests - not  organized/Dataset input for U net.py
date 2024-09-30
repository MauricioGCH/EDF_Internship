#Dataset input for U net

## Windows are going to be resized to 125x125


## If smaller, take more space, increase the windoz before cutting

## If bigger, just resize the cutted window

import os
import numpy as np
import cv2
import os
import glob
import pandas as pd

# Función para encontrar los límites de la máscara no cero
def get_bounding_box(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

# Directorio con las máscaras de segmentación

#C:\Users\d42684\Documents\STAGE\CODES\Small_ARIS_Mauzac


Train = glob.glob(os.path.join(r'C:\Users\d42684\Documents\STAGE\CODES\Small_ARIS_Mauzac\Train\2014*\Foreground\t*\*'))
Val = glob.glob(os.path.join(r'C:\Users\d42684\Documents\STAGE\CODES\Small_ARIS_Mauzac\Val\2014*\Foreground\t*\*'))
Test = glob.glob(os.path.join(r'C:\Users\d42684\Documents\STAGE\CODES\Small_ARIS_Mauzac\Test\2014*\Foreground\t*\*'))

AllImages = Train + Val + Test


# Lista para almacenar información de cada imagen
data = []

# Iterar sobre los archivos de imágenes encontrados
for filepath in AllImages:
    if filepath.endswith(('.png', '.jpg')):
        filename = os.path.basename(filepath)
        
        # Cargar la máscara
        mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        
        # Obtener los límites de la caja que contiene todos los valores no cero
        rmin, rmax, cmin, cmax = get_bounding_box(mask)
        
        # Calcular el tamaño del bounding box
        width = cmax - cmin + 1
        height = rmax - rmin + 1
        
        # Añadir la información al data
        data.append([filename, rmin, rmax, cmin, cmax, width, height])

# Crear un DataFrame de pandas
df = pd.DataFrame(data, columns=['filename', 'rmin', 'rmax', 'cmin', 'cmax', 'width', 'height'])

# Calcular promedio y mediana del tamaño del bounding box
average_width = df['width'].mean()
median_width = df['width'].median()
average_height = df['height'].mean()
median_height = df['height'].median()

# Crear DataFrames para promedio y mediana
average_df = pd.DataFrame([{'filename': 'average', 'width': average_width, 'height': average_height}])
median_df = pd.DataFrame([{'filename': 'median', 'width': median_width, 'height': median_height}])

# Concatenar los DataFrames
df = pd.concat([df, average_df, median_df], ignore_index=True)

# Guardar el DataFrame en un archivo CSV
df.to_csv(r'C:\Users\d42684\Documents\STAGE\CODES\U-Net\bounding_boxes.csv', index=False)

print(f"Promedio del ancho del bounding box: {average_width}")
print(f"Mediana del ancho del bounding box: {median_width}")
print(f"Promedio del alto del bounding box: {average_height}")
print(f"Mediana del alto del bounding box: {median_height}")