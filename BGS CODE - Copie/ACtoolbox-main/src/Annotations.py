import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.animation as animation
import ast
from skimage.measure import label,regionprops
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize
import json
import re
from test_smallfish_cleaning import filter_small_regions_by_size

def add_annotation(df, filename, file_size, object_count, comment, region_count, region_id, region_shape, region_attributes):
    """
    Adds a new annotation to the DataFrame.
    
    :param df: The DataFrame to add the annotation to.
    :param filename: Name of the file.
    :param file_size: Size of the file.
    :param object_count: Count of objects.
    :param region_count: Count of regions.
    :param region_id: ID of the region.
    :param region_shape: Shape attributes of the region (point or rect).
    :param region_attributes: Attributes of the region.
    :return: Updated DataFrame with the new annotation.
    """
    file_attributes = f'{{"Object_Count":"{object_count}", "Comment":"{comment}"}}'
    region_shape = json.loads(region_shape)
    if region_shape['name'] == 'point':
        region_shape_attributes = f'{{"name":"point","cx":{region_shape["cx"]},"cy":{region_shape["cy"]}}}'
    elif region_shape['name'] == 'rect':
        region_shape_attributes = f'{{"name":"rect","x":{region_shape["x"]},"y":{region_shape["y"]},"width":{region_shape["width"]},"height":{region_shape["height"]}}}'
    
    region_attributes = f'{{"Object":"{region_attributes}"}}'
    
    new_row = pd.DataFrame({
        'filename': [filename],
        'file_size': [file_size],
        'file_attributes': [file_attributes],
        'region_count': [region_count],
        'region_id': [region_id],
        'region_shape_attributes': [region_shape_attributes],
        'region_attributes': [region_attributes]
    })

    updated_df = pd.concat([df, new_row], ignore_index=True)
    
    #updated_df.to_csv(r'C:\Users\chapi\Documents\STAGE\CODE\bgslibrary\ACtoolbox-main\SecondVersion_Annotations_Filled.csv', index=False)
    return updated_df

# for the annotations of the subsense
def find_consecutive_sequences(df, col_name, sequence):
    seq_len = len(sequence)
    for start_idx in range(len(df) - seq_len + 1):
        if df[col_name].iloc[start_idx:start_idx + seq_len].tolist() == sequence:
            return df.iloc[start_idx:start_idx + seq_len]
        #else:
            #print("no match was found, review annotation and images")
            
    return pd.DataFrame(columns=df.columns)  # Return an empty DataFrame if no sequence is found

def is_centroid_inside_bbox(centroid, bbox, mask,frame):
    cx, cy = centroid
    x1, y1, x2, y2 = bbox
    
    print("Para el frame: ",frame," Imprimo el valor antes de la condicion : ",mask[cx,cy])
    if x1 <= cx <= x2 and y1 <= cy <= y2:
        print("Para el frame: ",frame, " el centroide esta adentro")
        if mask[cx,cy] != 0:
         print("Para el frame: ",frame, " Luego de confirmar que estq dentro, se confirmo el valor del centroid, que es : ",mask[cx,cy], " No toca recalcular")
         return True
        elif mask[cx,cy] == 0:
            print("Para el frame: ",frame, " Luego de confirmar que esta adentro, se visualiza el valor del centroid, que es : ",mask[cx,cy], " toca recalcular (deberia ser 0)")
            return False
    
    else:
        print("Para el frame: ",frame, " Luego de confirmar que no esta adentro, se visualiza el valor del centroid, que es : ",mask[cx,cy], " toca recalcular (deberia ser 0)")
        return False

def update_frame():
            ax1.imshow(AnnotatedImageListToVisualizePerTrack[frame_idx])
            ax2.imshow(OriginalImages[frame_idx])
            plt.pause(0.001)

def on_key(event):
            global frame_idx
            if event.key == 'right':  # 'n' key for next frame
                frame_idx = (frame_idx + 1) % len(AnnotatedImageListToVisualizePerTrack)
                update_frame()
                plt.draw()
            elif event.key == 'left':  # 'p' key for previous frame
                frame_idx = (frame_idx - 1) % len(AnnotatedImageListToVisualizePerTrack)
                update_frame()
                plt.draw()
            elif event.key == 'q':  # 'q' key to quit
                plt.close()

def reduce_bbox(bbox, percentage, center_point):
    """
    Reduces the size of the bounding box according to the given percentage, centered at the specified point.
    
    :param bbox: The bounding box in the format (xmin, ymin, xmax, ymax).
    :param percentage: The percentage by which to reduce the size of the bounding box.
    :param center_point: The point (x, y) around which to center the reduced bounding box.
    :return: The reduced bounding box in the format (xmin, ymin, xmax, ymax) as integers.
    """
    x_min, y_min, x_max, y_max = bbox
    center_x, center_y = center_point

    # Calculate the current width and height of the bbox
    current_width = x_max - x_min
    current_height = y_max - y_min

    # Calculate the new width and height based on the percentage
    new_area = (current_width * current_height) * (percentage)
    new_width = (new_area * current_width / current_height) ** 0.5
    new_height = (new_area * current_height / current_width) ** 0.5

    # Ensure the new bbox is centered around the given center_point
    new_x_min = center_x - new_width / 2
    new_x_max = center_x + new_width / 2
    new_y_min = center_y - new_height / 2
    new_y_max = center_y + new_height / 2

    return (int(new_x_min), int(new_y_min), int(new_x_max), int(new_y_max))

def calculate_medoid(mask, bboxOfRegion):
    """
    Calculate the medoid of an object in a mask.
    
    Parameters:
        mask (numpy.ndarray): A 2D array where the object pixels have value 255.
        
    Returns:
        tuple: Coordinates of the medoid (x, y).
    """
    #region = skeletonize(region)
    #region = np.where(region, 255, 0)
    ## get center of bbox to correctly place the mediod in the original mask
    x_min, y_min, x_max, y_max = bboxOfRegion
    
    CuttedRegion = mask[y_min:y_max,x_min:x_max]
    #plt.imshow(CuttedRegion, cmap = "gray")
    #plt.show()
    # Extract the coordinates of the object
    object_coords = np.argwhere(CuttedRegion == 255)
    
    if len(object_coords) == 0:
        raise ValueError("No object found in the mask with value 255.")
    
    # Calculate the pairwise distances
    distances = cdist(object_coords, object_coords, metric='euclidean')
    
    # Calculate the sum of distances for each point
    distance_sums = distances.sum(axis=1)
    
    # Find the index of the minimum sum
    medoid_index = np.argmin(distance_sums)
    
    # Get the medoid coordinates
    medoid_coords = object_coords[medoid_index]
    
    # place it in the whole mask coordinates, y, x is the format for cv2.circle
    medoid_coords = ( int(medoid_coords[1]+ x_min), int(medoid_coords[0] + y_min))
    return tuple(medoid_coords)

def get_fish_type():

    options = {
        1: "Eel",
        2: "Salmon",
        3: "CatFish",
        4: "SmallFish",
        5: "Trash",
        6: "Carp",
        7: "Trash_Arcing",
        8: "SmallFish_Arcing",
        9: "Eel_Arcing",
        10: "Salmon_Arcing",
        11: "CatFish_Arcing",
        12: "Carp_Arcing"
    }

    for key, value in options.items():
        print(f"{key}: {value}")

    while True:
        try:
            selection = int(input("Enter the number corresponding to your selection: "))
            if selection in options:
                return options[selection]
            else:
                print("Invalid selection. Please enter a number between 1 and 9.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

def get_sorted_image_paths(tracks_path, index_track):
    # Define the pattern to extract frame number
    pattern = re.compile(r"Binary_Obj_frame(\d+)\.png")
    
    # Get the list of image paths
    image_paths = glob.glob(os.path.join(tracks_path[index_track], "Binary_Obj_frame*.png"))
    
    # Sort the image paths based on the frame number
    sorted_image_paths = sorted(image_paths, key=lambda x: int(pattern.search(os.path.basename(x)).group(1)))
    
    return sorted_image_paths

#TODO LEER CARPETA DE RESULTADOS DE SubSENSE y mantener solo los videos no repetidos

#ListOfVideos = glob.glob(r'C:\Users\chapi\Documents\STAGE\CODE\bgslibrary\Test\ARIS_Mauzac_OtherFish\SuBSENSE\VideoModeResults\2014*')
ListOfVideos = glob.glob(r'C:\Users\chapi\Documents\STAGE\CODE\bgslibrary\Test\SELECTED\2014*')
ResultsToAnalyze = [] # Esto me toca cambiarlo luego ya que una vez agregue aunque sea una anotacion de un video, ya lo filtra

df = pd.read_csv(r'C:\Users\chapi\Documents\STAGE\CODE\bgslibrary\ACtoolbox-main\SecondVersion_Annotations_Filled.csv')

dfnames = list(set(df['filename'].tolist()))
Correctednames = []

for dfname in dfnames:
    name = dfname.split("_")[0] + "_" + dfname.split("_")[1]
    Correctednames.append(name)
Correctednames = list(set(Correctednames))

indexToDelete = []
for i in range(len(ListOfVideos)):
    videoName = ListOfVideos[i].split("\\")[-1]
    
    for correctedname in Correctednames:

        if correctedname == videoName:
            indexToDelete.append(i)

indexToDelete.sort(reverse=True)

for index in indexToDelete:
    if 0 <= index < len(ListOfVideos):  # Verifica que el índice sea válido
        del ListOfVideos[index]


print("wait")


# TODO Visualizar las imagenes con el bbox y centroide una por una como gif con el numero de la imagen. 
#(tiene que ser no muy rapido y que se repita hasta que presione espacio, ahi continua el codigo)

## En la visualizacion tengo que poner la image original al lado para poder comparar por ejemplo si el numero de individuos es correcto
for video in ListOfVideos:
    
    #"C:\Users\chapi\Documents\STAGE\CODE\bgslibrary\Test\ARIS_Mauzac\SuBSENSE\VideoModeResults\2014-11-06_043000\Track_final_Images\Bbox_labels.xlsx"
    videoName = video.split("\\")[-1]
    AnnotationsPath_xlsx = os.path.join(video,r'Track_final_Images\Bbox_labels.xlsx')
    Annotations = pd.read_excel(AnnotationsPath_xlsx)

    TracksPath = glob.glob(os.path.join(video,r'Track_final_Images\Foreground\track*'))
    OriginalTracksPath = glob.glob(os.path.join(video,r'Track_final_Images\Original\track*'))
    print("RIGHT NOW WE ARE ON VIDEO:n ", videoName)


    
    for indexTrack in range(len(TracksPath)) : 
        trackNumber = TracksPath[indexTrack].split("\\")[-1].split("k")[-1]
        Fg_points = [] #for the end to add
        bboxs = [] #for the end to add
        ImageNames = [] #for the end
        #images_path = glob.glob(os.path.join(TracksPath[indexTrack],"Binary_Obj_frame*.png"))
        images_path = get_sorted_image_paths(TracksPath, indexTrack)

        original_images_path = glob.glob(os.path.join(OriginalTracksPath[indexTrack],"Obj_frame*.jpg"))
            # to match with the annotation
        ##cv2.imread(path)
        #cv2.cvtColor(Forvideo, cv2.COLOR_GRAY2BGR)

        OriginalImages =[]
        OriginalImagesToSave = [] # Tomode selected tracks and save them with the formatted name
        MaskToSave = []
        ImageListToVisualizePerTrack = [] 
        ImgNumberPerTrack = []

        AnnotatedImageListToVisualizePerTrack = []

        for indexImage in range(len(images_path)): 
            ##Las imagenes las tengo que cargar en color para agregar las bbox y centroide
            OriginalImages.append(cv2.imread(original_images_path[indexImage]))
            OriginalImagesToSave.append(cv2.imread(original_images_path[indexImage]))

            #original image name format for .csv file
            ImageNames.append(str(videoName + "_" + "t" + trackNumber + "_" +images_path[indexImage].split("\\")[-1].split(".")[0].split("_")[1]
                                  +"_"+images_path[indexImage].split("\\")[-1].split(".")[0].split("_")[2] + ".jpg")) # Ex : '2014-11-06_043000'  + "_" + "t" + trackNumber + "_"  Obj_frame4098 + .jpg

            ImgNumberPerTrack.append(int(images_path[indexImage].split("\\")[-1].split(".")[0].split("e")[1]))

            image = cv2.imread(images_path[indexImage])
            
            ImageListToVisualizePerTrack.append(image)

        TrackAnnotation_df = find_consecutive_sequences(Annotations, "Frame", ImgNumberPerTrack)
        print(TrackAnnotation_df)
        if len(TrackAnnotation_df) > len(ImgNumberPerTrack):
            input("Rare case where two tracks have the exact same frames, stop and review case !")
            #print("The frame is ", frame)
        




        for index, row in TrackAnnotation_df.iterrows():
            bbox = ast.literal_eval(row['bbox'])
            bboxs.append(bbox)
            try :
                centroid = ast.literal_eval(row['centroid'])
            except (ValueError, SyntaxError):
                if pd.isna(row['centroid']):
                    centroid = (0, 0)
                else:
                    raise
            frame = row['Frame']
            image_to_modify = ImageListToVisualizePerTrack[index - TrackAnnotation_df.index[0]].copy() # For videos that have several tracks substract TrackAnnotation_df.index[0]
           
            
            
            cv2.circle(image_to_modify,centroid,5,(255,0,0),-1)
            #cv2.circle(OriginalImages[index - TrackAnnotation_df.index[0]],centroid,5,(255,0,0),-1)
            
            cv2.rectangle(image_to_modify,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(255,0,0),3)
            cv2.putText(image_to_modify, str(frame), (50, int(image_to_modify.shape[0]*0.9)) , cv2.FONT_HERSHEY_SIMPLEX ,2, (0, 0, 255) , 4, cv2.LINE_AA)
            AnnotatedImageListToVisualizePerTrack.append(image_to_modify)
            
            
            #cv2.rectangle(OriginalImages[index - TrackAnnotation_df.index[0]],(bbox[0],bbox[1]),(bbox[2],bbox[3]),(255,0,0),3)
            #cv2.putText(OriginalImages[index - TrackAnnotation_df.index[0]], str(frame), (50, int(image_to_modify.shape[0]*0.9)) , cv2.FONT_HERSHEY_SIMPLEX ,2, (0, 0, 255) , 4, cv2.LINE_AA)
        


            
            
        assert len(AnnotatedImageListToVisualizePerTrack) == len(OriginalImages)

        frames = []
        fig, (ax1, ax2) = plt.subplots(1, 2)
        title_obj = fig.suptitle('')
        
        for img1, img2, title in zip(AnnotatedImageListToVisualizePerTrack, OriginalImages, ImgNumberPerTrack):
            frame1 = ax1.imshow(img1, animated=True)
            frame2 = ax2.imshow(img2, animated=True)
            title_obj.set_text("Track Analysis")
            frames.append([frame1, frame2])
        

        ani = animation.ArtistAnimation(fig, frames, interval=350, blit=True, repeat=True)

        ax1.axis('off')
        ax2.axis('off')
        plt.show()



        Add_Or_Not = input(" Will this track be used/added to the annotations ? (press y for YES, n for NO)")

        if Add_Or_Not == "n": #skip current track
            
            continue 
        
        
        #if selected_fish_type == "SmallFish_Arcing":

        Smallfisharcing = str(input(" Case of SmallFish with arcing (1) or Smallfish(2) with other regions where only one fish is tracked ? No (3) "))

        AnnotatedImageListToVisualizePerTrack = []
        dict_for_Frames = {}

        if Smallfisharcing == "1" or Smallfisharcing == "2": #skip current track

            for index, row in TrackAnnotation_df.iterrows():
                bboxs = []
                Fg_points = []
                #bbox = ast.literal_eval(row['bbox'])
                #bboxs.append(bbox)
                
                frame = row['Frame']
                image_to_modify = ImageListToVisualizePerTrack[index - TrackAnnotation_df.index[0]] # For videos that have several tracks substract TrackAnnotation_df.index[0]

                
                

                #TODO agregar recalculo del centroide para el caso necesario, se determina necesario con is_centroid_inside_bbox, falta la funcion para recalcularlo
                TheMaskToCompare = cv2.cvtColor(image_to_modify, cv2.COLOR_BGR2GRAY)
                TheMaskToCompare =  np.where(TheMaskToCompare != 0, 255, 0)

                if Smallfisharcing == "1":
                    TheMaskToCompare = filter_small_regions_by_size(TheMaskToCompare)
                    image_to_modify = filter_small_regions_by_size(image_to_modify)
                else:
                    TheMaskToCompare = filter_small_regions_by_size(TheMaskToCompare, threshold_percentage= 0.10)
                    image_to_modify = filter_small_regions_by_size(image_to_modify, threshold_percentage= 0.10)

                #plt.imshow(TheMaskToCompare, cmap="gray")
                #plt.show()

                MaskToSave.append(TheMaskToCompare)
                #print(bbox)
                #plt.imshow(TheMaskToCompare, cmap = "gray")
                #plt.show()

                #medoid = calculate_medoid(TheMaskToCompare, bbox)
                #Fg_points.append(medoid)

                props = regionprops(label(TheMaskToCompare))
                AnnotatedImageListToVisualizePerTrack.append(image_to_modify)
                for prop in props:
                    bboxs.append((prop.bbox[1],prop.bbox[0],prop.bbox[3],prop.bbox[2]))
                    Fg_points.append((int(prop.centroid[1]),int(prop.centroid[0])))


                    cv2.circle(image_to_modify,(int(prop.centroid[1]),int(prop.centroid[0])),5,(255,0,0),-1)
                    cv2.circle(OriginalImages[index - TrackAnnotation_df.index[0]],(int(prop.centroid[1]),int(prop.centroid[0])),5,(255,0,0),-1)
                    
                    cv2.rectangle(image_to_modify,(prop.bbox[1],prop.bbox[0]),(prop.bbox[3],prop.bbox[2]),(255,0,0),3)
                    cv2.putText(image_to_modify, str(frame), (50, int(image_to_modify.shape[0]*0.9)) , cv2.FONT_HERSHEY_SIMPLEX ,2, (0, 0, 255) , 4, cv2.LINE_AA)
                    
                    
                    
                    cv2.rectangle(OriginalImages[index - TrackAnnotation_df.index[0]],(prop.bbox[1],prop.bbox[0]),(prop.bbox[3],prop.bbox[2]),(255,0,0),3)
                    cv2.putText(OriginalImages[index - TrackAnnotation_df.index[0]], str(frame), (50, int(image_to_modify.shape[0]*0.9)) , cv2.FONT_HERSHEY_SIMPLEX ,2, (0, 0, 255) , 4, cv2.LINE_AA)
                
                dict_for_Frames[frame] = (bboxs,Fg_points)#

        
        elif Smallfisharcing == "3":

        
        
            for index, row in TrackAnnotation_df.iterrows():
                bboxs = []
                Fg_points = []
                bbox = ast.literal_eval(row['bbox'])
                bboxs.append(bbox)
                
                frame = row['Frame']
                image_to_modify = ImageListToVisualizePerTrack[index - TrackAnnotation_df.index[0]] # For videos that have several tracks substract TrackAnnotation_df.index[0]

                
                

                #TODO agregar recalculo del centroide para el caso necesario, se determina necesario con is_centroid_inside_bbox, falta la funcion para recalcularlo
                TheMaskToCompare = cv2.cvtColor(image_to_modify, cv2.COLOR_BGR2GRAY)
                TheMaskToCompare =  np.where(TheMaskToCompare != 0, 255, 0)
                MaskToSave.append(TheMaskToCompare)
                #print(bbox)
                #plt.imshow(TheMaskToCompare, cmap = "gray")
                #plt.show()

                medoid = calculate_medoid(TheMaskToCompare, bbox)
                Fg_points.append(medoid)

                dict_for_Frames[frame] = (bboxs,Fg_points)
                
                
                # dentro = is_centroid_inside_bbox(centroid, bbox, TheMaskToCompare,frame)
                # if dentro:

                #     cv2.circle(image_to_modify,centroid,5,(255,0,0),-1)
                #     cv2.circle(OriginalImages[index],centroid,5,(255,0,0),-1)

                # else:
                    
                #      centroid = validate_centroid(bbox, TheMaskToCompare)
                #      print("New centroid is:  ", centroid)
                #      print("Valor del pixel en el nuevo centroide : ", TheMaskToCompare[centroid[0],centroid[1]])
                #      if TheMaskToCompare[centroid[0],centroid[1]] != 0:
                #         valuecount = valuecount + 1
                #      cv2.circle(image_to_modify,centroid,5,(255,0,0),-1)
                #      cv2.circle(OriginalImages[index],centroid,5,(255,0,0),-1)
                #      count = count + 1

                

                # centroid = validate_centroid(bbox, TheMaskToCompare)
                # cv2.circle(image_to_modify,(centroid[1],centroid[0]),5,(255,0,0),-1)
                # cv2.circle(OriginalImages[index],(centroid[1],centroid[0]),5,(255,0,0),-1)
                # cv2.circle(image_to_modify,centroid,5,(255,0,0),-1)
                # cv2.circle(OriginalImages[index],centroid,5,(255,0,0),-1)
                
                #cv2.putText(imgExplanation_to_write, 'Frame #'+str(frame), (frame_width, 60) , cv2.FONT_HERSHEY_SIMPLEX ,2, (0, 0, 255) , 5, cv2.LINE_AA) 

                cv2.circle(image_to_modify,medoid,5,(255,0,0),-1)
                cv2.circle(OriginalImages[index - TrackAnnotation_df.index[0]],medoid,5,(255,0,0),-1)
                
                cv2.rectangle(image_to_modify,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(255,0,0),3)
                cv2.putText(image_to_modify, str(frame), (50, int(image_to_modify.shape[0]*0.9)) , cv2.FONT_HERSHEY_SIMPLEX ,2, (0, 0, 255) , 4, cv2.LINE_AA)
                AnnotatedImageListToVisualizePerTrack.append(image_to_modify)
                
                
                cv2.rectangle(OriginalImages[index - TrackAnnotation_df.index[0]],(bbox[0],bbox[1]),(bbox[2],bbox[3]),(255,0,0),3)
                cv2.putText(OriginalImages[index - TrackAnnotation_df.index[0]], str(frame), (50, int(image_to_modify.shape[0]*0.9)) , cv2.FONT_HERSHEY_SIMPLEX ,2, (0, 0, 255) , 4, cv2.LINE_AA)
            

        
            
            
        assert len(AnnotatedImageListToVisualizePerTrack) == len(OriginalImages)

        frames = []
        fig, (ax1, ax2) = plt.subplots(1, 2)
        title_obj = fig.suptitle('')
        
        for img1, img2, title in zip(AnnotatedImageListToVisualizePerTrack, OriginalImages, ImgNumberPerTrack):
            frame1 = ax1.imshow(img1, animated=True)
            frame2 = ax2.imshow(img2, animated=True)
            title_obj.set_text("Track Analysis")
            frames.append([frame1, frame2])
        

        ani = animation.ArtistAnimation(fig, frames, interval=350, blit=True, repeat=True)

        ax1.axis('off')
        ax2.axis('off')
        plt.show()


        
        
        # if they will be added i will save the original image in the name format
        os.makedirs("SELECTED ORIGINAL IMAGES", exist_ok= True)
        for i in range(len(ImageNames)):
            cv2.imwrite(os.path.join("SELECTED ORIGINAL IMAGES",ImageNames[i]), OriginalImagesToSave[i])
            cv2.imwrite(os.path.join("SELECTED ORIGINAL IMAGES",str("m_"+ ImageNames[i].split(".")[0] + ".png")), MaskToSave[i])
            #TheMaskToCompare

# -_-_-_-_-_-_-_--_-_-_-_-_-_-_-_-__--_-_---_
        # # Create the figure and subplots
        # fig, (ax1, ax2) = plt.subplots(1, 2)

        # # Create title for the entire figure
        # title_obj = fig.suptitle('')

        # # Set initial frame index
        # frame_idx = 0

        # # Update the initial frame
        # update_frame()

        # # Remove axis for better visualization
        # ax1.axis('off')
        # ax2.axis('off')

        # # Connect key press event to the function
        # fig.canvas.mpl_connect('key_press_event', on_key)

        # # Show the plot
        # plt.show()

        #TODO Al haber visualizado el resultado debo haber 
        """"Clase del objeto, 
	    como tratar los prompts (si la anguila esta fragmentada (este caso sirve tambien para los demas dado como utilizamos SAM despues) o tiene arcing, son dos caminos 	diferentes), 
	    Comentario a agregar sobre la calidad del prompt/dificulta del track.
        """
        
        selected_fish_type = get_fish_type() # String


        


        comentarios_por_frame = {} #DICCIONARIO CON LOS COMENTARIOS POR FRAME

        while True:
            # Solicitar al usuario que ingrese el número de frame
            frame = input("Please for comments, enter the number fo the frame (or write 'exit' to finish): ")
            
            # Verificar si el usuario quiere terminar
            if frame.lower() == 'exit':
                break
            
            # Solicitar al usuario que ingrese el comentario
            comentario = input(f"Write your comment for frame {frame}: ")
            
            # Guardar el comentario en el diccionario
            comentarios_por_frame[frame] = comentario
            
            # Confirmar que el comentario ha sido guardado
            print(f"Comment saved for frame {frame}.\n")

        # Mostrar todos los comentarios guardados
        print("Saved commentaries:")
        for frame, comentario in comentarios_por_frame.items():
            print(f"Frame {frame}: {comentario}")

        HowToTreat = int(input("To treat as a a general case (Also works for fragmented eel) enter 1, to treat as an object with Arcing press 2"))
        #df = []
        #df= pd.read_csv(r"C:\Users\chapi\Documents\STAGE\CODE\segment-anything-main\notebooks\SecondVersion_Annotations_Filled.csv") 
        if HowToTreat == 1:
            #Fragmented Eel
            filenames = ImageNames
            file_size = 0
            object_count = 1 # TODO count of the track, modify later 
            comment = comentarios_por_frame
            region_count = 0 #TODO to see what to do with this, not used but VIA might use it
            region_id = 0 #TODO to see what to do with this, not used but VIA might use it


            



            region_attributes = selected_fish_type #ex: Eel

            for i in range(len(ImgNumberPerTrack)):

                if str(ImgNumberPerTrack[i]) in comment:
                    frame_comment = comment[str(ImgNumberPerTrack[i])]
                else : frame_comment = ''
                
                box, points = dict_for_Frames[ImgNumberPerTrack[i]]


                Fg_points_json = []
                bboxs_json = []

                for j in range(len(box)):
                    Fg_points_json.append(f'{{"name":"point","cx":{points[j][0]},"cy":{points[j][1]}}}')

                    #width = xmax - xmin
                    #height = ymax - ymin
                    bboxs_json.append(f'{{"name":"rect","x":{box[j][0]},"y":{box[j][1]},"width":{int(box[j][2]-box[j][0])},"height":{int(box[j][3]-box[j][1])}}}')


                for k in range(len(bboxs_json)):
                    df = add_annotation(df=df, filename=filenames[i], file_size=file_size, ## for points
                                object_count=object_count, comment=frame_comment, 
                                region_count=region_count, region_id=region_id, 
                                region_shape=Fg_points_json[k], region_attributes=region_attributes ) #este i de iterar no deberia ser el mismop, toca crear otro loop
                    
                    df = add_annotation(df=df, filename=filenames[i], file_size=file_size, ##for bboxs
                                object_count=object_count, comment=frame_comment, 
                                region_count=region_count, region_id=region_id, 
                                region_shape=bboxs_json[k], region_attributes=region_attributes )

                
        elif HowToTreat == 2 :
            # Object with arcing
            
            df= df
            filenames = ImageNames
            file_size = 0
            object_count = 1
            comment = comentarios_por_frame
            region_count = 0 #TODO to see what to do with this, not used but VIA might use it
            region_id = 0 #TODO to see what to do with this, not used but VIA might use it


            # Fg_points_json = []
            # bboxs_json = []
            # for i in range(len(Fg_points)):
            #     Fg_points_json.append(f'{{"name":"point","cx":{Fg_points[i][0]},"cy":{Fg_points[i][1]}}}')

            #     #width = xmax - xmin
            #     #height = ymax - ymin
            #     reduced_bbox = reduce_bbox(bboxs[i], 0.2, Fg_points[i]) # 0.2 is the take percentage of the are so it is reduced by 0.8
            #     bboxs_json.append(f'{{"name":"rect","x":{reduced_bbox[0]},"y":{reduced_bbox[1]},"width":{int(reduced_bbox[2]-reduced_bbox[0])},"height":{int(reduced_bbox[3]-reduced_bbox[1])}}}')



            region_attributes = selected_fish_type #ex: Eel

            for i in range(len(ImgNumberPerTrack)):
                if str(ImgNumberPerTrack[i]) in comment:
                    frame_comment = comment[str(ImgNumberPerTrack[i])]
                else : frame_comment = ''

                box, points = dict_for_Frames[ImgNumberPerTrack[i]]


                Fg_points_json = []
                bboxs_json = []

                for j in range(len(box)):
                    Fg_points_json.append(f'{{"name":"point","cx":{points[j][0]},"cy":{points[j][1]}}}')

                    #width = xmax - xmin
                    #height = ymax - ymin
                    reduced_bbox = reduce_bbox(box[j], 0.2, points[j]) # 0.2 is the take percentage of the are so it is reduced by 0.8
                    bboxs_json.append(f'{{"name":"rect","x":{reduced_bbox[0]},"y":{reduced_bbox[1]},"width":{int(reduced_bbox[2]-reduced_bbox[0])},"height":{int(reduced_bbox[3]-reduced_bbox[1])}}}')
                
                for k in range(len(bboxs_json)):
                    df = add_annotation(df=df, filename=filenames[i], file_size=file_size, ## for points
                                object_count=object_count, comment=frame_comment, 
                                region_count=region_count, region_id=region_id, 
                                region_shape=Fg_points_json[k], region_attributes=region_attributes )
                    
                    df = add_annotation(df=df, filename=filenames[i], file_size=file_size, ##for bboxs
                                object_count=object_count, comment=frame_comment, 
                                region_count=region_count, region_id=region_id, 
                                region_shape=bboxs_json[k], region_attributes=region_attributes )
        
        df.to_csv(r'C:\Users\chapi\Documents\STAGE\CODE\bgslibrary\ACtoolbox-main\SecondVersion_Annotations_Filled.csv', index=False)

#--------------------------------------------------------------------------------------------------------


#TODO Al haber dado toda esa informacion, de manera automatica se verifica si hay centroide, si esta dentro de la bbox, y si el valor en la mascara es de objeto(hay problemas con los centroides)
        #TODO Si toca determionar un nuevo centroide, mediante una funcion que utiliza la bbox, primero verificar que haya una sola region, 
        #y entonces obtener entonces el centroide con regionrpops(label()).centroide
            #TODO para la anguila ademas del centroide deberia agregar otros dos puntos en las esquinas a lo largo

#TODO Al haber ya determinado todo la info agregar la nueva fila, me toca