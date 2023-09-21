import numpy as np
import tensorflow as tf
from data_processing import get_image_with_glomeruli, get_image_with_no_glomeruli, resize_images, shuffle

epoch = 10
batch_size = 8
path_dataset = '../../dataset512x512/image_train_augmented_balanced.npy'
path_label = '../../dataset512x512/label_train_augmented_balanced.npy'


def load_data():
    ds = np.load(path_dataset)
    lb = np.load(path_label)
    return ds, lb


def preprocess_data(ds, lb):
    glomeruli = get_image_with_glomeruli(ds, lb)
    glomeruli = resize_images(glomeruli)
    glomeruli_labels = np.ones(glomeruli.shape[0])

    no_glomeruli = get_image_with_no_glomeruli(ds, lb)
    no_glomeruli = resize_images(no_glomeruli)
    no_glomeruli_labels = np.zeros(no_glomeruli.shape[0])

    dataset = np.append(glomeruli, no_glomeruli, axis=0)
    labels = np.append(glomeruli_labels, no_glomeruli_labels, axis=0)

    dataset, labels = shuffle(dataset, labels)

    return dataset, labels


def data_generator(ds, lb):
    print('Minimo train:', ds.min())
    print('Massimo train:', ds.max())
    print('Data shape:', ds.shape)

    for d, l in zip(ds, lb):
        yield (d, l)


def process_data(image, label):
    return tf.cast(image, tf.float32)/255, label


def load_generator(ds, lb):
    output_shapes_train = (tf.TensorSpec(shape=(200, 200, 3), dtype=tf.float32),
                           tf.TensorSpec(shape=(), dtype=tf.int64))

    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(ds, lb), output_signature=output_shapes_train)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.map(
        process_data, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

def generate_dataset():
    ds, lb = load_data()
    ds, lb = preprocess_data(ds, lb)
    dataset = load_generator(ds, lb)
    return dataset

def tuning(model, dataset):
    x = model(model.input)
    x = tf.keras.layers.Dense(4096, activation="relu")(x)
    x = tf.keras.layers.Dense(4096, activation="relu")(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
    model = tf.keras.Model(model.input, outputs)

    for i, layer in enumerate(model.layers):
       if i > 10:
           layer.trainable = True
       else:
           layer.trainable = False

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("Fine-tuning model")
    # finetune the model
    model.fit(dataset, epochs=epoch, batch_size=batch_size)

    layer_to_remove = model.layers[-4].name
    # Create a new model with the outputs of the original model
    feature_extractor = tf.keras.models.Model(
        inputs=model.input, outputs=model.get_layer(layer_to_remove).output)
    #print(feature_extractor.summary())

    return model, feature_extractor
