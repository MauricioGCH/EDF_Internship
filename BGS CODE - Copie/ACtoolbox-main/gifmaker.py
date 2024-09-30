from PIL import Image
import imageio
import os
import cv2
import numpy as np
import glob

def create_gif(image_folder, output_path, fps):
    # List all image files in the specified folder
    images = []
    for file_name in os.listdir(image_folder):
        if file_name.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            file_path = os.path.join(image_folder, file_name)
            images.append(file_path)
    
    # Sort the images by name to maintain a sequence
    images.sort()

    # Load images using Pillow
    frames = [Image.open(image) for image in images]

    # Convert frames to the format required by imageio
    frames = [frame.convert('RGBA') for frame in frames]
    frame_duration = 1 / fps  # Duration of each frame in seconds

    # Save as a GIF
    imageio.mimsave(output_path, frames, format='GIF', duration=frame_duration)


def images_to_gif(image_list, output_path, duration=0.5):
    """
    Convierte una lista de imágenes en un GIF.
    
    :param image_list: Lista de imágenes.
    :param output_path: Ruta del archivo de salida para el GIF.
    :param duration: Duración de cada frame en segundos.
    """
    with imageio.get_writer(output_path, mode='I', duration=duration) as writer:
        for image in image_list:
            # Asegurarse de que la imagen está en formato RGB
            if len(image.shape) == 2 or image.shape[2] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            writer.append_data(image)


def images_to_video(image_list, output_path, fps=30, frame_size=None):
    """
    Convierte una lista de imágenes en un video.

    :param image_list: Lista de imágenes (matrices numpy).
    :param output_path: Ruta donde se guardará el video.
    :param fps: Cuadros por segundo del video.
    :param frame_size: Tamaño del cuadro (ancho, alto). Si es None, se tomará el tamaño de la primera imagen.
    """
    if not image_list:
        print("La lista de imágenes está vacía.")
        return

    # Usar la primera imagen para obtener el tamaño del cuadro si no se proporciona
    first_image = image_list[0]
    if frame_size is None:
        frame_size = (first_image.shape[1], first_image.shape[0])  # (ancho, alto)

    # Definir el codec y crear el objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Puedes usar otros codecs como 'XVID', 'MJPG', etc.
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    for img in image_list:
        if img is None:
            print(f"No se pudo procesar la imagen: {img}")
            continue
        # Redimensionar la imagen si el tamaño del cuadro no coincide
        if (img.shape[1], img.shape[0]) != frame_size:
            img = cv2.resize(img, frame_size)
        video_writer.write(img)

    video_writer.release()
    print(f"Video guardado en: {output_path}")

def draw_mask_edges(image, mask, color=(0, 255, 0), thickness=1):
    """
    Dibuja los bordes de la máscara sobre la imagen original.
    
    :param image: Imagen original en formato RGB.
    :param mask: Máscara binaria de la imagen.
    :param color: Color de los bordes (en formato BGR).
    :param thickness: Grosor de los bordes.
    :return: Imagen con los bordes de la máscara dibujados.
    """
    # Encontrar contornos en la máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Dibujar los contornos en la imagen original
    result = image.copy()
    cv2.drawContours(result, contours, -1, color, thickness)
    
    return result

# Cargar la imagen y la máscara

masks_path = glob.glob(r"C:\Users\chapi\Documents\STAGE\rESULTADOS PARA 12-06 fontainebleau\2014-11-14_190000\sam\t0\m_Eel0_2014-11-14_190000_t0_Obj_frame*.jpg")

image_paths = glob.glob(r"C:\Users\chapi\Documents\STAGE\rESULTADOS PARA 12-06 fontainebleau\2014-11-14_190000\bg\Original\track0\Obj_frame*.jpg")

ImageList =[]
for i in range(len(masks_path)):

    image = cv2.imread(image_paths[i])
    mask = cv2.imread(masks_path[i], cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    #print(np.unique(mask))


    result = draw_mask_edges(image, mask)
    ImageList.append(result)
    #cv2.imshow('Mask Edges', result)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
# Mostrar el resultado



output_gif_path = "2014-11-14_190000_SAM.mp4"
#images_to_gif(ImageList, output_gif_path, duration=0.5)
images_to_video(ImageList, output_gif_path, fps=7)


