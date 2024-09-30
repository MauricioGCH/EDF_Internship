#prueba

import numpy as np
from scipy.spatial import distance_matrix
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from skimage.morphology import binary_closing, skeletonize
from scipy.ndimage import label, find_objects


def find_furthest_points(binary_image):
    """
    Encuentra los puntos más distantes dentro de un objeto en una imagen binaria.
    
    Parameters:
        binary_image (numpy.ndarray): Imagen binaria con un solo objeto (255) en un fondo (0).
        
    Returns:
        tuple: Coordenadas de los dos puntos más distantes y la distancia entre ellos.
    """
    # Extraer las coordenadas de los píxeles del objeto
    object_coords = np.argwhere(binary_image == 255)
    
    # Calcular la matriz de distancias entre todos los puntos del objeto
    dist_matrix = distance_matrix(object_coords, object_coords)
    
    # Encontrar los dos puntos con la mayor distancia entre ellos
    i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    return object_coords[i], object_coords[j], dist_matrix[i, j]

def find_three_points(binary_image, max_distance):
    """
    Encuentra tres puntos dentro del objeto con las condiciones especificadas.
    
    Parameters:
        binary_image (numpy.ndarray): Imagen binaria con un solo objeto (255) en un fondo (0).
        max_distance (float): La distancia máxima entre los dos puntos más lejanos del objeto.
        
    Returns:
        list: Lista con las coordenadas de los tres puntos.
    """
    # Extraer las coordenadas de los píxeles del objeto
    object_coords = np.argwhere(binary_image == 255)
    
    # Calcular la matriz de distancias entre todos los puntos del objeto
    dist_matrix = distance_matrix(object_coords, object_coords)
    
    target_distance_70 = 0.95 * max_distance
    target_distance_35 = 0.475 * max_distance
    found_points = None
    
    # Encontrar dos puntos que estén a una distancia aproximada del 70% de la distancia máxima
    for i in range(len(object_coords)):
        for j in range(i + 1, len(object_coords)):
            if np.isclose(dist_matrix[i, j], target_distance_70, rtol=0.1):
                point1, point2 = object_coords[i], object_coords[j]
                found_points = [point1, point2]
                break
        if found_points:
            break
    
    if not found_points:
        raise ValueError("No se encontraron dos puntos que cumplan con el criterio de distancia.")
    
    # Encontrar un tercer punto que esté a una distancia aproximada del 35% de la distancia máxima entre los dos puntos anteriores
    for k in range(len(object_coords)):
        if np.isclose(np.linalg.norm(object_coords[k] - point1), target_distance_35, rtol=0.2) and \
           np.isclose(np.linalg.norm(object_coords[k] - point2), target_distance_35, rtol=0.2):
            found_points.append(object_coords[k])
            break
    
    if len(found_points) == 3:
        return found_points
    else:
        raise ValueError("No se encontró un tercer punto que cumpla con el criterio de distancia.")

def main(binary_image):
    """
    Encuentra y verifica los tres puntos específicos dentro del objeto en una imagen binaria.
    
    Parameters:
        binary_image (numpy.ndarray): Imagen binaria con un solo objeto (255) en un fondo (0).
        
    Returns:
        list: Lista con las coordenadas de los tres puntos verificados.
    """
    # Encontrar los puntos más lejanos y la distancia entre ellos
    point1, point2, max_distance = find_furthest_points(binary_image)
    print(f"Puntos más lejanos: {point1}, {point2} con una distancia de {max_distance}")
    
    # Encontrar tres puntos dentro del objeto que cumplan con el criterio de distancia
    three_points = find_three_points(binary_image, max_distance)
    print(f"Tres puntos encontrados: {three_points}")
    
    # Verificar que los puntos estén dentro del objeto
    for point in three_points:
        if binary_image[tuple(point)] != 255:
            raise ValueError(f"El punto {point} no está dentro del objeto (valor no es 255).")
    
    return three_points


# --------------------------------------------------------------



def analyze_gaps(binary_image):
    grayscale_image = binary_image
    # Convert the input binary image to grayscale
    #grayscale_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    # Ensure the image is in the correct format (8-bit single-channel)
    if grayscale_image.dtype != np.uint8:
        grayscale_image = grayscale_image.astype(np.uint8)
    # Use distance transform to get the distance map
    dist_transform = cv2.distanceTransform(grayscale_image, cv2.DIST_L2, 5)
    # Normalize the distance map
    dist_transform = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
    return dist_transform

def apply_closing(binary_image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    return closed_image

def evaluate_connected_components(image):
    # Ensure the input image is of the correct type (8-bit unsigned)
    image = cv2.convertScaleAbs(image)
    # Find all connected components
    num_labels, labels_im = cv2.connectedComponents(image)
    return num_labels, labels_im

def remove_distant_connections(labels_im, max_distance):
    # Label the connected components
    labeled_array, num_features = label(labels_im)
    regions = find_objects(labeled_array)
    
    # Create a copy of the labeled image to modify
    filtered_labels = np.copy(labels_im)
    
    # Iterate through the regions and calculate distances
    for i, region in enumerate(regions):
        for j in range(i+1, len(regions)):
            region1 = regions[i]
            region2 = regions[j]
            
            # Calculate the minimum distance between the two regions
            r1_centroid = [(region1[0].start + region1[0].stop) / 2, (region1[1].start + region1[1].stop) / 2]
            r2_centroid = [(region2[0].start + region2[0].stop) / 2, (region2[1].start + region2[1].stop) / 2]
            
            distance = np.sqrt((r1_centroid[0] - r2_centroid[0])**2 + (r1_centroid[1] - r2_centroid[1])**2)
            
            if distance > max_distance:
                # If the distance is greater than the max_distance, separate the regions
                filtered_labels[labeled_array == i+1] = 0
                filtered_labels[labeled_array == j+1] = 0
    
    # Create a binary image from the filtered labels
    filtered_image = np.where(filtered_labels > 0, 255, 0).astype(np.uint8)
    
    return filtered_image

def adaptive_morphological_closing(binary_image, max_distance):
    dist_transform = analyze_gaps(binary_image)
    gap_size = int(np.percentile(dist_transform, 95) * 10)  # Adjust multiplier as needed
    
    best_num_labels = float('inf')
    best_closed_image = binary_image
    
    # Try different kernel sizes
    for kernel_size in range(1, gap_size + 1, 2):  # Use odd sizes only
        closed_image = apply_closing(binary_image, kernel_size)
        num_labels, labels_im = evaluate_connected_components(closed_image)
        
        if num_labels < best_num_labels:
            best_num_labels = num_labels
            best_closed_image = closed_image
    
    # Post-process to remove connections based on distance
    _, labels_im = evaluate_connected_components(best_closed_image)
    filtered_image = remove_distant_connections(labels_im, max_distance)
    
    return filtered_image

# Ejemplo de uso
#"C:\Users\chapi\Documents\STAGE\CODE\bgslibrary\Test\ARIS_Mauzac\SuBSENSE\VideoModeResultsOld\2014-11-06_043000\Track_final_Images\Foreground\track0\Binary_Obj_frame4117.jpg"
#"C:\Users\chapi\Documents\STAGE\CODE\bgslibrary\Test\ARIS_Mauzac\SuBSENSE\VideoModeResultsOld\2014-11-06_043000\Track_final_Images\Foreground\track0\Binary_Obj_frame4162.jpg"
#"C:\Users\chapi\Documents\STAGE\CODE\bgslibrary\Test\ARIS_Mauzac\SuBSENSE\VideoModeResultsOld\2014-11-06_043000\Track_final_Images\Foreground\track0\Binary_Obj_frame4113.jpg"
binary_image = cv2.imread(r"C:\Users\chapi\Documents\STAGE\CODE\bgslibrary\Test\ARIS_Mauzac\SuBSENSE\VideoModeResultsOld\2014-11-06_043000\Track_final_Images\Foreground\track0\Binary_Obj_frame4158.jpg", cv2.IMREAD_GRAYSCALE)
binary_image =  np.where(binary_image != 255, 0, 255)

plt.title('reformating Binary Image')
plt.imshow(binary_image, cmap='gray')

plt.show()
binary_image = ndimage.binary_fill_holes(binary_image).astype(int)
binary_image = binary_closing(binary_image, footprint=np.ones((3,3)))
binary_image =  np.where(binary_image == 1, 255, 0)

plt.title('filling holes Binary Image')
plt.imshow(binary_image, cmap='gray')

plt.show()


max_distance = 200  # Adjust the distance threshold as needed
closed_image = adaptive_morphological_closing(binary_image, max_distance)

plt.title('connection Binary Image')
plt.imshow(closed_image, cmap='gray')

plt.show()

skeleton = skeletonize(binary_image)
skeleton =  np.where(skeleton == 1, 255, 0)


three_points = main(skeleton)
print("Coordenadas de los tres puntos:", tuple(three_points[0]),tuple(three_points[1]), tuple(three_points[2]))

for i in range(len(three_points)):
    point = tuple(three_points[i])

    binary_image[point[0], point[1]] = 127


plt.imshow(binary_image,cmap="gray")
plt.show()




