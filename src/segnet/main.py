import math
import numpy as np
from segnet import SegNet
import tensorflow as tf
import json

image_train_path = '../../slides/image_train_augmented.npy'
label_train_path = '../../annotations/label_train_augmented.npy'
image_validation_path = '../../slides/image_validation.npy'
label_validation_path = '../../annotations/label_validation.npy'
image_test_path = '../../slides/image_test.npy'
label_test_path = '../../annotations/label_test.npy'
version = 1.0
path_to_save_model = f'../../saved_model/segnet_V{version}.h5'
path_to_save_history = f'../shistory/history_V{version}.json'
epochs = 50
learning_rates = [0.001, 0.01, 0.1]
batch_size = 32  # non utilizzato


strategy = tf.distribute.MirroredStrategy()

with strategy.scope():

    image_train = np.load(image_train_path)
    label_train = np.load(label_train_path)
    image_validation = np.load(image_validation_path)
    label_validation = np.load(label_validation_path)
    # image_test = np.load(image_test_path)
    # label_test = np.load(label_test_path)

    # validation_data = generate_data_tensor(image_validation, label_validation)
    # test_data = generate_data_tensor(image_test, image_test, train=False)

    # Definisci la funzione per addestrare e valutare un modello con un dato tasso di apprendimento
    def train_and_evaluate_segnet(image_train, label_train, validation_image, validation_label, learning_rate):
        # Costruisci il modello
        model = SegNet()
        # Compila il modello con la funzione di perdita e l'ottimizzatore appropriati
        model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=["accuracy", "Precision", "Recall", "FalseNegatives",
                                                                                                                   "FalsePositives", "TrueNegatives", "TruePositives"])
        print(f"params: {model.count_params()}")
        print("testing a model")
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, verbose=1, mode='max', restore_best_weights=True)

        #model.fit(image_train, label_train, epochs=epochs)
        history = model.fit(image_train, label_train, validation_data=(image_validation, label_validation), epochs=epochs, callbacks=[early_stopping])

        # Valuta il modello sul validation set
        #evals = model.evaluate(validation_image, validation_label)
        best_accuracy = max(history.history['val_accuracy'])
        #return model, evals[1]
        return model, best_accuracy, history

    best_accuracy = 0.0
    best_learning_rate = None
    best_models = None
    best_history = None

    # Valuta ogni tasso di apprendimento e seleziona il migliore
    for learning_rate in learning_rates:
        print("creating a model")
        model, accuracy, history = train_and_evaluate_segnet(
            image_train, label_train, image_validation, label_validation, learning_rate)
        print(f"Learning Rate: {learning_rate}, Accuracy: {accuracy}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_learning_rate = learning_rate
            best_model = model
            best_history = history

    best_models.save(path_to_save_model)
    with open(path_to_save_history, "w") as fp:
        json.dump(best_history.history, fp)
