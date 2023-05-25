import threading
import openslide
from utils import extrapolate_patches
import numpy as np


def process_svs_file(svs_file, path_to_annotations, path_to_images, el_width, el_height, output_width, output_height):
    thread_name = threading.current_thread().name
    file = open("../log/thread_" + thread_name + ".txt", "x")
    file.write("Sono il thread" + thread_name + "\n")
    file.write("Sto elaborando il file " +
               svs_file[len(path_to_images):-4] + "\n")
    file.close()
    # Ottieni il percorso del file .xml corrispondente
    annotation = path_to_annotations + \
        svs_file[len(path_to_images):-4] + ".xml"
    # Carica l'immagine svs
    wsi = openslide.OpenSlide(svs_file)
    d, l = extrapolate_patches(
        wsi, annotation, el_width, el_height, output_width, output_height)
    np.save('../slides/' +
            svs_file[len(path_to_images):-4] + '.npy', np.array(d))
    np.save('../annotations/' +
            svs_file[len(path_to_images):-4] + '_label.npy', np.array(l))
    return d, l