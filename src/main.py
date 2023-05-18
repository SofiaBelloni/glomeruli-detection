import glob
import openslide
import cv2
from matplotlib import pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
from utils import extrapolate_patches

# import tensorflow as tf

path_to_images = "../slides/"
path_to_annotations = "../annotations/"
output_width = 2000
output_height = 2000

dataset = []
labels = []

# Ottieni la lista dei file .svs nella cartella slides
svs_files = glob.glob(path_to_images + "*.svs")

for svs_file in svs_files:
    # Ottieni il percorso del file .xml corrispondente
    annotation = path_to_annotations + \
        svs_file[len(path_to_images):-4] + ".xml"
    # Carica l'immagine svs
    wsi = openslide.OpenSlide(svs_file)
    d, l = extrapolate_patches(wsi, annotation, output_width, output_height)
    dataset.extend(d)
    labels.extend(l)

dataset = np.array(dataset)
labels = np.array(labels)

print(dataset.shape)

# for i in range(len(dataset)):
#  plt_image(dataset[i], labels[i])

np.save('dataset.npy', dataset)
