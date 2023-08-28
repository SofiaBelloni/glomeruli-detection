import math
import numpy as np
from unet import Unet
import tensorflow as tf
import json

# image_train_path = '../../dataset512x512/image_train_augmented.npy'
# label_train_path = '../../dataset512x512/label_train_augmented.npy'
image_train_path = '../../dataset512x512/image_train.npy'
label_train_path = '../../dataset512x512/label_train.npy'
image_validation_path = '../../dataset512x512/image_validation.npy'
label_validation_path = '../../dataset512x512/label_validation.npy'
image_test_path = '../../dataset512x512/image_test.npy'
label_test_path = '../../dataset512x512/label_test.npy'
version = 1.1
path_to_save_model = f'../../saved_model/unet_V{version}'
path_to_save_model_weights = f'../../saved_model/unet_V{version}_weights.h5'
path_to_save_history = f'../../histories/history_V{version}.json'
path_to_save_test_result = f'../../dataset512x512/test_result_unet_V{version}.npy'
epochs = 50
# learning_rates = [0.001, 0.01, 0.1]
learning_rate = 0.01
batch_size = 8
use_generator = True
use_weights = True


strategy = tf.distribute.MirroredStrategy()

with strategy.scope():

    def train_and_evaluate_unet(image_train=None, label_train=None, validation_image=None, validation_label=None, dataset_train=None, dataset_validation=None, learning_rate=None):
        # Costruisci il modello
        model = Unet()
        # Compila il modello con la funzione di perdita e l'ottimizzatore appropriati
        if use_weights:
            model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=["accuracy", "Precision", "Recall", "FalseNegatives",
                                                                                                                       "FalsePositives", "TrueNegatives", "TruePositives"], sample_weight_mode='temporal')
        else:
            model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=["accuracy", "Precision", "Recall", "FalseNegatives",
                                                                                                                       "FalsePositives", "TrueNegatives", "TruePositives"])
        # print(f"params: {model.count_params()}")
        print("testing a model")
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=3, verbose=1, mode='max', restore_best_weights=True)

        # model.fit(image_train, label_train, epochs=epochs)
        # history = model.fit(image_train, label_train, validation_data=(image_validation, label_validation), epochs=epochs, callbacks=[early_stopping])
        if image_train is not None and label_train is not None:
            history = model.fit(image_train, label_train, validation_data=(
                validation_image, validation_label), epochs=epochs, batch_size=batch_size)
        elif dataset_train is not None and dataset_validation is not None:
            history = model.fit(
                dataset_train, validation_data=dataset_validation, epochs=epochs)
        # Valuta il modello sul validation set
        # evals = model.evaluate(validation_image, validation_label)
        best_accuracy = max(history.history['val_accuracy'])
        # return model, evals[1]
        return model, best_accuracy, history

    def data_generator(class_weights=None):
        data = np.load(image_train_path)
        labels = np.load(label_train_path)

        print('Minimo train:', data.min())
        print('Massimo train:', data.max())
        print('Data shape:', data.shape)

        for d, l in zip(data, labels):
            if class_weights is None:
                yield d, l
            else:
                sample_weights = np.zeros_like(labels, dtype=np.float32)
                sample_weights[l == 1] = class_weights[1]  # "glomerulo"
                # "non glomerulo"
                sample_weights[l == 0] = class_weights[0]
                yield (d, l, sample_weights)

    def validation_data_generator():
        validation_data = np.load(image_validation_path)
        validation_labels = np.load(label_validation_path)

        print('Minimo validation:', validation_data.min())
        print('Massimo validation:', validation_data.max())
        print('Validation data shape:', validation_data.shape)

        for d, l in zip(validation_data, validation_labels):
            yield d, l

    def process_data(image, label):
        return tf.cast(image, tf.float32)/255, tf.one_hot(label, 2, name="label", axis=-1)

    def load_generator():
        class_weights = None
        if use_weights:
            class_weights = compute_class_weights()

        output_shapes = (tf.TensorSpec(shape=(512, 512, 3), dtype=tf.float32),
                         tf.TensorSpec(shape=(512, 512), dtype=tf.int64))
        train_gen = data_generator(class_weights)
        dataset = tf.data.Dataset.from_generator(
            train_gen, output_signature=output_shapes)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.map(
            process_data, num_parallel_calls=tf.data.AUTOTUNE)

        validation_dataset = tf.data.Dataset.from_generator(
            validation_data_generator, output_signature=output_shapes)
        validation_dataset = validation_dataset.batch(
            batch_size, drop_remainder=True)
        validation_dataset = validation_dataset.map(
            process_data, num_parallel_calls=tf.data.AUTOTUNE)

        return dataset, validation_dataset

    def load_data():
        image_train = np.load(image_train_path)
        label_train = np.load(label_train_path)
        image_validation = np.load(image_validation_path)
        label_validation = np.load(label_validation_path)
        return image_train, label_train, image_validation, label_validation

    def test_model(model, image_test):
        return model.predict(image_test)

    def load_test_data():
        image_test = np.load(image_test_path)/255
        return image_test

    def compute_class_weights():
        labels = np.load(label_train_path)
        # Count the number of samples for each class
        class_0_count = np.sum(labels == 0)
        class_1_count = np.sum(labels == 1)
        total_samples = labels.size  # Total number of pixels

        # Calculate class weights using the formula
        class_0_weight = total_samples / (2 * class_0_count)
        class_1_weight = total_samples / (2 * class_1_count)

        del labels

        return {0: class_0_weight, 1: class_1_weight}

    # best_accuracy = 0.0
    # best_learning_rate = None
    # best_models = None
    # best_history = None

    # image_test = np.load(image_test_path)
    # label_test = np.load(label_test_path)

    # validation_data = generate_data_tensor(image_validation, label_validation)
    # test_data = generate_data_tensor(image_test, image_test, train=False)

    # Definisci la funzione per addestrare e valutare un modello con un dato tasso di apprendimento

    # Valuta ogni tasso di apprendimento e seleziona il migliore
    # for learning_rate in learning_rates:
    print("creating a model")
    if (use_generator):
        dataset, validation_dataset = load_generator()
        model, accuracy, history = train_and_evaluate_unet(dataset_train=dataset, dataset_validation=validation_dataset,
                                                           learning_rate=learning_rate)
    else:
        image_train, label_train, image_validation, label_validation = load_data()
        model, accuracy, history = train_and_evaluate_unet(
            image_train=image_train, label_train=label_train, validation_image=image_validation, validation_label=label_validation, learning_rate=learning_rate)
    print(f"Learning Rate: {learning_rate}, Accuracy: {accuracy}")
    # if accuracy > best_accuracy:
    #     best_accuracy = accuracy
    #     best_learning_rate = learning_rate
    #     best_model = model
    #     best_history = history

    model.save(path_to_save_model, save_format='tf')
    model.save_weights(path_to_save_model_weights)

    with open(path_to_save_history, "w") as fp:
        json.dump(history.history, fp)

    image_test = load_test_data()
    test_model(model, image_test)
