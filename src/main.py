import concurrent.futures
import threading
import glob
import cv2
from matplotlib import pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
from utils import extrapolate_patches
import tensorflow as tf

# The path can also be read from a config file, etc.
import os
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    OPENSLIDE_PATH = r'C:\Users\sofia\openslide-win64-20230414\openslide-win64-20230414\bin'
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

path_to_images = "../slides/"
path_to_annotations = "../annotations/"
el_width = 2000
el_height = 2000
output_width = 400
output_height = 400

dataset = []
labels = []

file = open("../log/job.txt", "x")
file.write("Il task è iniziato\n")
file.close()


# Ottieni la lista dei file .svs nella cartella slides
svs_files = glob.glob(path_to_images + "*.svs")


def process_svs_file(svs_file):
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
    print("wth?")
    d, l = extrapolate_patches(
        wsi, annotation, el_width, el_height, output_width, output_height)
    np.save('../slides/' +
            svs_file[len(path_to_images):-4] + '.npy', np.array(d))
    np.save('../annotations/' +
            svs_file[len(path_to_images):-4] + '_label.npy', np.array(l))
    return d, l


# Creazione di un ThreadPoolExecutor con un numero di thread desiderato
num_threads = 9  # Numero di thread da utilizzare
executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)

# Lista per salvare i future restituiti dalle chiamate asincrone
futures = []
file = open("../log/job.txt", "a")
file.write("Lancio i thread\n")

# Esecuzione della funzione extrapolate_patches in parallelo per ogni svs_file
for svs_file in svs_files:
    future = executor.submit(process_svs_file, svs_file)
    futures.append(future)

file.write("Aspetto la fine dei thread\n")
# Attendere il completamento di tutte le chiamate asincrone
concurrent.futures.wait(futures)

file.write("I thread hanno finito, concateno i risultati\n")
# Ottenere i risultati dai future
dataset = []
labels = []
for future in futures:
    d, l = future.result()
    dataset.extend(d)
    labels.extend(l)

dataset = np.array(dataset)
labels = np.array(labels)

np.save('../slides/dataset.npy', dataset)
np.save('../annotations/labels.npy', labels)

file.write("Risultati salvati\n")

file.close()
