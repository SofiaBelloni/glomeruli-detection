import cv2
from matplotlib import pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
import threading


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


def extrapolate_patches(wsi, annotation, el_width, el_height, output_width, output_height):
    # Ottieni le dimensioni dell'immagine
    w, h = wsi.dimensions
    label_image = np.zeros((h, w), dtype=np.uint8)
    annotations = get_annotatios(annotation)
    get_labels(label_image, annotations)

    # Calcola il numero di righe e colonne necessarie per suddividere l'immagine
    num_rows = h // el_height
    num_cols = w // el_width

    # Crea un'immagine di output con le stesse dimensioni dell'immagine svs

    dataset = []
    labels = []

    thread_name = threading.current_thread().name
    file = open("../log/thread_" + thread_name + ".txt", "a")
    file.write("Sto per leggere il file wsi\n")

    wsi = np.array(wsi.read_region((0, 0), 0, (w, h)))

    file.write("File letto\n")
    for row in range(num_rows):
        for col in range(num_cols):
            # for row in range(3, 5):
            #    for col in range(58, 60):
            # Calcola le coordinate di inizio e fine per l'immagine corrente
            x = col * el_width
            y = row * el_height
            x_end = x + el_width
            y_end = y + el_height

            # Estrai l'immagine corrente
            region = wsi[y: y_end, x: x_end]
            image = cv2.cvtColor(region, cv2.COLOR_RGBA2BGR)

            is_white, p = is_mostly_white(image)
            if not is_white:
                r_image = cv2.resize(
                    image, (output_width, output_height), interpolation=cv2.INTER_CUBIC)
                r_label_image = cv2.resize(
                    label_image[y:y_end, x: x_end], (output_width, output_height), interpolation=cv2.INTER_CUBIC)

                dataset.append(r_image)
                labels.append(r_label_image)
                if not ((col == num_cols-1) or (row == num_rows-1)):
                    x_h = x + el_width // 2
                    x_v = x
                    x_d = x + el_width // 2
                    y_h = y
                    y_v = y + el_height // 2
                    y_d = y + el_height // 2
                    region_h = wsi[y_h: y_h + el_height,
                                   x_h: x_h + el_width]
                    region_v = wsi[y_v: y_v + el_height,
                                   x_v: x_v + el_width]
                    region_d = wsi[y_d: y_d + el_height,
                                   x_d: x_d + el_width]
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
                        r_image = cv2.resize(
                            image_h, (output_width, output_height), interpolation=cv2.INTER_CUBIC)
                        r_label_image = cv2.resize(
                            label_image[y_h: y_h+el_height, x_h: x_h+el_width], (output_width, output_height), interpolation=cv2.INTER_CUBIC)

                        dataset.append(r_image)
                        labels.append(r_label_image)

                    if not is_white_v:
                        r_image = cv2.resize(
                            image_v, (output_width, output_height), interpolation=cv2.INTER_CUBIC)
                        r_label_image = cv2.resize(
                            label_image[y_v: y_v+el_height, x_v: x_v+el_width], (output_width, output_height), interpolation=cv2.INTER_CUBIC)

                        dataset.append(r_image)
                        labels.append(r_label_image)

                    if not is_white_d:
                        r_image = cv2.resize(
                            image_d, (output_width, output_height), interpolation=cv2.INTER_CUBIC)
                        r_label_image = cv2.resize(
                            label_image[y_d: y_d+el_height, x_d: x_d+el_width], (output_width, output_height), interpolation=cv2.INTER_CUBIC)

                        dataset.append(r_image)
                        labels.append(r_label_image)

    file.write("Wsi elaborato\nDataset di dimensione:" +
               str(np.array(dataset).shape) + "\nLabels di dimensione:" + str(np.array(labels).shape))
    file.close()
    return dataset, labels
