#visualization de resultados de sam


import os
import glob
from collections import defaultdict

def organize_tracks(base_path):
    # Crear un diccionario para almacenar las rutas de los archivos organizadas por objetos rastreados
    tracks = defaultdict(lambda: defaultdict(list))

    # Recorrer todas las carpetas en el directorio base
    for folder in glob.glob(os.path.join(base_path, "2014*")):
        if os.path.isdir(folder):
            # Obtener la fecha, hora y pista del nombre de la carpeta
            folder_name = os.path.basename(folder)
            parts = folder_name.split('_', 3)
            if len(parts) == 4:
                date, time, track, frame = parts
                track_id = f"{date}_{time}_{track}"

                # Buscar todos los archivos dentro de la carpeta
                for file in glob.glob(os.path.join(folder, "*")):
                    if os.path.isfile(file):
                        # Obtener el nombre del archivo
                        file_name = os.path.basename(file)
                        
                        # Extraer el tipo de objeto rastreado del nombre del archivo
                        object_type = file_name[2:-38]
                        
                        # Usar el track_id como clave y agregar la ruta del archivo a la lista correspondiente
                        tracks[track_id][object_type].append(file)
            else:
                print(f"Skipping folder with unexpected format: {folder_name}")

    return tracks
# m_SmallFish_Arcing0_2014-11-05_184000_t0_Obj_frame314
# Usar la función para organizar los tracks
base_path = r"C:\Users\chapi\Documents\STAGE\CODE\bgslibrary\ACtoolbox-main\Dataset\Small_ARIS_Mauzac\TEST\All_Originals\NewMasks"
organized_tracks = organize_tracks(base_path)

# Imprimir los resultados organizados
for track_id, objects in organized_tracks.items():
    print(f"{track_id}:")
    for object_type, paths in objects.items():
        print(f"  {object_type}:")
        for path in paths:
            print(f"    {path}")

# len(organized_tracks["2014-11-08_060000_t0"]['Eel_Arcing0']) THIS WORKS FOR RESULTS THAT ONLY HAVE ONE OBJECT PER TRACK; EITHER WAY I NEED TO FIX THE RESULTS FROM THE FIRST PART AS THEY HAVE DIFFERENT IDS