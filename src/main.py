import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from data_prepr_aug import generate_data_tensor
from segnet import SegNet
import tensorflow as tf
from unet import Unet
import json
import concurrent.futures
from thread import data_aug_thread

# The path can also be read from a config file, etc.
import os
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    OPENSLIDE_PATH = r'C:\Users\sofia\openslide-win64-20230414\openslide-win64-20230414\bin'
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

#tf.compat.v1.disable_eager_execution()

path_to_dataset = "../dataset512x512/dataset.npy"
path_to_labels = "../dataset512x512/labels.npy"

file = open("../log/training.txt", "x")

file.write("Loading dataset and labels")

dataset = np.load(path_to_dataset)
labels = np.load(path_to_labels)

file.write(
    f"Dataset and labels loaded\nDataset shape {dataset.shape} \nLabels shape {labels.shape}")

image_train, image_vt, label_train, label_vt = train_test_split(
    dataset, labels, test_size=0.30, random_state=42)
image_validation, image_test, label_validation, label_test = train_test_split(
    image_vt, label_vt, test_size=0.33, random_state=42)

file.write("Dataset and labels splitted in train and test set\n" +
      f"image_train shape {image_train.shape} - label_train shape {label_train.shape}" +
      f"image_test shape {image_test.shape} - image_test shape {label_test.shape}")


num_threads = 4  # Numero di thread da utilizzare
executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)
futures = []

for i in range(0,4):
    future = executor.submit(data_aug_thread, image_train, label_train, image_train[0].shape)
    futures.append(future)
    
concurrent.futures.wait(futures)

for future in futures:
    im, l = future.result()
    image_train=np.append(image_train,im)
    label_train=np.append(label_train,l)
    
    
np.save('../slides/image_train_augmented.npy', image_train)
np.save('../annotations/label_train_augmented.npy', label_train)

file.write("Applied data agumentation to train set\n" +
      f"image_train augmented shape {image_train.shape} - label_train augmented shape {label_train.shape}")


train_data = generate_data_tensor(image_train, label_train)
validation_data = generate_data_tensor(image_validation, label_validation)
test_data = generate_data_tensor(image_test, image_test, train=False)

file.write("Preprocessed and created tensor dataset")




steps_per_epoch = len(image_train) // 128 




# Definisci la funzione per addestrare e valutare un modello con un dato tasso di apprendimento
def train_and_evaluate_segnet(train_data, validation_data, learning_rate, steps_per_epoch):
    # Costruisci il modello
    model = SegNet()
    # Compila il modello con la funzione di perdita e l'ottimizzatore appropriati
    model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=["accuracy", "Precision", "Recall", "FalseNegatives",
                                                                                                               "FalsePositives", "TrueNegatives", "TruePositives"])
    print("testing a model")
    # Addestra il modello sul training set
    
    

    history = model.fit(train_data, epochs=50, steps_per_epoch=steps_per_epoch)
    # Valuta il modello sul validation set
    evals = model.evaluate(validation_data)
    return model, evals[1], history
def train_and_evaluate_unet(train_data, validation_data, learning_rate, steps_per_epoch):
    # Costruisci il modello
    model = Unet()
    # Compila il modello con la funzione di perdita e l'ottimizzatore appropriati
    model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=["accuracy", "Precision", "Recall", "FalseNegatives",
                                                                                                               "FalsePositives", "TrueNegatives", "TruePositives"])
    print("testing a model")
    # Addestra il modello sul training set
    history = model.fit(train_data, epochs=50, steps_per_epoch=steps_per_epoch)
    # Valuta il modello sul validation set
    evals = model.evaluate(validation_data)
    return model, evals[1], history

best_accuracy = 0.0
best_learning_rate = None
best_models = None
best_history = None

learning_rates = [0.001, 0.01, 0.1]

file.write("Start training on segnet")

# Valuta ogni tasso di apprendimento e seleziona il migliore
for learning_rate in learning_rates:
    print("creating a model")
    model, accuracy, history = train_and_evaluate_segnet(train_data, validation_data, learning_rate, steps_per_epoch)
    print(f"Learning Rate: {learning_rate}, Accuracy: {accuracy}")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_learning_rate = learning_rate
        best_model = model
        best_history = history

best_models.save('../saved_model/segnet')
file.write(f"best accuracy {best_accuracy}\n best learning rate {best_learning_rate}")
with open("../shistory/history_segnet.txt", "x") as fp:
    json.dump(best_history.history, fp)

file.write("Start training on unet")

best_accuracy = 0.0
best_learning_rate = None
best_models = None
best_history = None

for learning_rate in learning_rates:
    print("creating a model")
    model, accuracy, history = train_and_evaluate_unet(train_data, validation_data, learning_rate, steps_per_epoch)
    print(f"Learning Rate: {learning_rate}, Accuracy: {accuracy}")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_learning_rate = learning_rate
        best_model = model
        best_history = history
        
best_models.save('../saved_model/unet')
with open("../uhistory/history_unet.txt", "x") as fp:
    json.dump(best_history.history, fp)
file.write(f"best accuracy {best_accuracy}\n best learning rate {best_learning_rate}")

file.close()
