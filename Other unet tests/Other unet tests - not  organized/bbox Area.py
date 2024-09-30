    ## Finding the input size for UNET
import pandas as pd
import json

# Leer el archivo CSV
df = pd.read_csv(r"SecondVersion_Annotations_Filled.csv")

# Función para calcular el área de la bbox
def calculate_area(region_shape_attributes):
    try:
        region = json.loads(region_shape_attributes)
        if region.get("name") == "rect":
            return region["width"] * region["height"]
        else:
            return 0
    except json.JSONDecodeError:
        return 0

def is_eel(region_attributes):
    try:
        attributes = json.loads(region_attributes)
        return attributes.get('Object') == "SmallFish"
    except json.JSONDecodeError:
        return False
# Añadir una nueva columna para el área de la bbox
df['bbox_area'] = df['region_shape_attributes'].apply(calculate_area)

# Filtrar las filas que tienen "name":"rect" y ordenar por el área de la bbox
rect_df = df[(df['bbox_area'] > 0) & (df['region_attributes'].apply(is_eel))]

rect_df = rect_df.sort_values(by='bbox_area', ascending=False)



# Seleccionar las 10 filas con la bbox más grande
top_10_rects = rect_df.head(10)
print("the biggest eel is: ",top_10_rects["region_shape_attributes"].iloc[0])
#print(top_10_rects)