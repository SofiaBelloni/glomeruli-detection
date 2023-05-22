import cv2
from matplotlib import pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET


def get_annotatios(file_path):
    # Parsa il file XML delle annotazioni
    tree = ET.parse(file_path)
    root = tree.getroot()
    # Ottieni tutte le annotazioni dal file XML
    annotations = []
    for annotation in root.iter('Annotation'):
        name = annotation.get('Name')
        coordinates = []
        for coordinate in annotation.iter('Coordinate'):
            x = float(coordinate.get('X'))
            y = float(coordinate.get('Y'))
            coordinates.append((x, y))
        annotations.append({'name': name, 'coordinates': coordinates})
    return annotations


def is_mostly_white(image, threshold_w=0.85, threshold_p=0.98):
    # Converti l'immagine in scala di grigi
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calcola la soglia per considerare i pixel bianchi
    pixel_threshold = int(threshold_w * 255)

    # Conta i pixel bianchi nell'immagine
    white_pixels = np.sum(gray_image >= pixel_threshold)

    # Calcola la percentuale di pixel bianchi rispetto alla dimensione totale dell'immagine
    white_percentage = white_pixels / \
        (gray_image.shape[0] * gray_image.shape[1])

    # Verifica se la percentuale di pixel bianchi supera la soglia
    if white_percentage >= threshold_p:
        return True, white_percentage
    else:
        return False, white_percentage


def get_labels(labels, annotations):
    for annotation in annotations:
        polygon = np.array([annotation['coordinates']], dtype=np.int32)
        cv2.fillPoly(labels, polygon, 1)


def plt_image(image, labes):
    fig, axs = plt.subplots(1, 2)
    # Primo subplot: labels
    axs[0].imshow(labes)
    axs[0].axis('off')
    # Secondo subplot: immagine
    axs[1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[1].axis('off')
    # Mostra i subplot affiancati
    plt.show()


def extrapolate_patches(wsi, annotation, output_width, output_height):
    # Ottieni le dimensioni dell'immagine
    w, h = wsi.dimensions
    # Calcola il numero di righe e colonne necessarie per suddividere l'immagine
    num_rows = h // output_height
    num_cols = w // output_width

    annotations = get_annotatios(annotation)
    # Crea un'immagine di output con le stesse dimensioni dell'immagine svs
    label_image = np.zeros((h, w), dtype=np.uint8)
    get_labels(label_image, annotations)

    dataset = []
    labels = []

    for row in range(num_rows):
        for col in range(num_cols):
    #for row in range(3, 5):
    #    for col in range(58, 60):
            # Calcola le coordinate di inizio e fine per l'immagine corrente
            x = col * output_width
            y = row * output_height
            x_end = x + output_width
            y_end = y + output_height

            # Estrai l'immagine corrente
            region = wsi.read_region(
                (x, y), 0, (output_width, output_height))
            image = cv2.cvtColor(np.array(region), cv2.COLOR_RGBA2BGR)

            is_white, p = is_mostly_white(image)
            if not is_white:
                dataset.append(image)
                labels.append(label_image[y:y_end, x: x_end])
                if not ((col == num_cols-1) or (row == num_rows-1)):
                    x_h = x + output_width // 2
                    x_v = x
                    x_d = x + output_width // 2
                    y_h = y
                    y_v = y + output_height // 2
                    y_d = y + output_width // 2
                    region_h = wsi.read_region(
                        (x_h, y_h), 0, (output_width, output_height))
                    region_v = wsi.read_region(
                        (x_v, y_v), 0, (output_width, output_height))
                    region_d = wsi.read_region(
                        (x_d, y_d), 0, (output_width, output_height))
                    image_h = cv2.cvtColor(
                        np.array(region_h), cv2.COLOR_RGBA2BGR)
                    image_v = cv2.cvtColor(
                        np.array(region_v), cv2.COLOR_RGBA2BGR)
                    image_d = cv2.cvtColor(
                        np.array(region_d), cv2.COLOR_RGBA2BGR)
                    is_white_h, _ = is_mostly_white(image_h)
                    is_white_v, _ = is_mostly_white(image_v)
                    is_white_d, _ = is_mostly_white(image_d)
                    if not is_white_h:
                        dataset.append(image_h)
                        labels.append(
                            label_image[y_h: y_h+output_height, x_h: x_h+output_width])
                    if not is_white_v:
                        dataset.append(image_v)
                        labels.append(
                            label_image[y_v: y_v+output_height, x_v: x_v+output_width])
                    if not is_white_d:
                        dataset.append(image_d)
                        labels.append(
                            label_image[y_d: y_d+output_height, x_d: x_d+output_width])
    return dataset, labels
