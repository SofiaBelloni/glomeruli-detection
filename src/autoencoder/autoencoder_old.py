import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from keras.layers import Dense, Flatten, Reshape, Input, InputLayer, Conv2D, Conv2DTranspose, MaxPool2D
from keras.models import Sequential, Model


def glomeruli_crop(glomeruli, glomeruli_labels):
    result = []
    original_images = []
    original_labels = []
    result_labels = []

    for j in range(0, len(glomeruli)):
        num_labels, labels = cv2.connectedComponents(
            glomeruli_labels[j], connectivity=8)
        for i in range(1, num_labels):
            image = glomeruli[j]

            # Find row and column indices of pixels with value 1
            yrow, xcol = np.where(labels == i)

            # Calculate minimum and maximum row indices
            yrowmin = np.min(yrow)
            yrowmax = np.max(yrow)

            # Calculate minimum and maximum column indices
            xcolmin = np.min(xcol)
            xcolmax = np.max(xcol)

            # Calculate the size of the side of the square as the max between the height and the width
            side = max(yrowmax - yrowmin + 1, xcolmax - xcolmin + 1)
            if (side >= 20):
                # Adjust the bounding box to be square
                yrowmin = max(
                    0, yrowmin - (side - (yrowmax - yrowmin + 1)) // 2)
                yrowmax = yrowmin + side
                if yrowmax > image.shape[0]:
                    yrowmax = image.shape[0]
                    yrowmin = yrowmax - side
                xcolmin = max(
                    0, xcolmin - (side - (xcolmax - xcolmin + 1)) // 2)
                xcolmax = xcolmin + side
                if xcolmax > image.shape[1]:
                    xcolmax = image.shape[1]
                    xcolmin = xcolmax - side

                # Extract the square image from the original image
                cropped_image = image[yrowmin:yrowmax, xcolmin:xcolmax]

                # Calculate the proportion of glomerulus pixels in the cropped image
                cropped_label = labels[yrowmin:yrowmax, xcolmin:xcolmax] == i
                glomerulus_proportion = np.mean(cropped_label)

                # Only keep the image if the glomerulus proportion is above the threshold
                if glomerulus_proportion >= 0.6:
                    # Resize image
                    resized_image = cv2.resize(
                        cropped_image, (200, 200), interpolation=cv2.INTER_AREA)
                    cropped_label = labels[yrowmin:yrowmax, xcolmin:xcolmax]
                    resized_label = cv2.resize(
                        cropped_label, (200, 200), interpolation=cv2.INTER_NEAREST)

                    result.append(resized_image)
                    result_labels.append(resized_label)
                    original_images.append(image)
                    original_labels.append(labels)

    return np.array(result), np.array(result_labels), np.array(original_images), np.array(original_labels)


def process_data(image):
    return tf.cast(image, tf.float32)/255, tf.cast(image, tf.float32)/255

# path_to_dataset = "../../dataset512x512/dataset.npy"
# path_to_labels = "../../dataset512x512/labels.npy"
#
# dataset = np.load(path_to_dataset)
# labels = np.load(path_to_labels)
# Identification of images and labels with glomerulus
# glomeruli = np.array([dataset[i] for i in range(len(dataset)) if (1 in labels[i])])
# glomeruli_labels = np.array([labels[i] for i in range(len(dataset)) if (1 in labels[i])])


# result, result_label, ori, ori_l = glomeruli_crop(glomeruli, glomeruli_labels)
#
# result = result.astype('float32') / 255

glomeruli_data = np.load("../../dataset512x512/glomeruli_data_no_dataaug.npy")

X_train, X_vt = train_test_split(
    glomeruli_data, test_size=0.3, random_state=42)

X_test, X_validation = train_test_split(
    X_vt, test_size=0.33, random_state=42)


def build_autoencoder(img_shape, code_size):
    encoder = Sequential()
    encoder.add(InputLayer(img_shape))
    encoder.add(Flatten())
    encoder.add(Dense(code_size))

    decoder = Sequential()
    decoder.add(InputLayer((code_size,)))
    decoder.add(Dense(np.prod(img_shape)))
    decoder.add(Reshape(img_shape))


def build_autoencoder_v3(img_shape, code_size):
    encoder = Sequential()
    encoder.add(InputLayer(img_shape))
    encoder.add(Conv2D(code_size//4, 3, padding="same"))
    encoder.add(MaxPool2D(2))
    encoder.add(Conv2D(code_size//2, 3, padding="same"))
    encoder.add(MaxPool2D(2))
    encoder.add(Conv2D(code_size, 3, padding="same"))
    encoder.add(MaxPool2D(2))

    decoder = Sequential()
    decoder.add(Conv2DTranspose(code_size, 5, padding="same", strides=2))
    decoder.add(Conv2DTranspose(code_size//2, 3, padding="same", strides=2))
    decoder.add(Conv2DTranspose(code_size//4, 2, padding="same", strides=2))
    decoder.add(Conv2DTranspose(3, 3, padding="same"))
    return encoder, decoder


def build_autoencoder_dense(img_shape, code_size):

    # 8 = 2*2*2 da MaxPool2D
    pre_flatten_shape = (img_shape[0] // 8, img_shape[1] // 8, code_size)

    encoder = Sequential()
    encoder.add(InputLayer(img_shape))
    encoder.add(Conv2D(code_size//4, 3, padding="same"))
    encoder.add(MaxPool2D(2))
    encoder.add(Conv2D(code_size//2, 3, padding="same"))
    encoder.add(MaxPool2D(2))
    encoder.add(Conv2D(code_size, 3, padding="same"))
    encoder.add(MaxPool2D(2))

    encoder.add(Flatten())
    encoder.add(Dense(1024, activation='relu'))

    decoder = Sequential()
    decoder.add(InputLayer((1024,)))
    # Mappa lo spazio latente di 1024 alla dimensione pre_flatten
    decoder.add(Dense(np.prod(pre_flatten_shape), activation='relu'))
    # Riorganizza i dati nella forma originale
    decoder.add(Reshape(pre_flatten_shape))

    decoder.add(Conv2DTranspose(code_size, 5, padding="same", strides=2))
    decoder.add(Conv2DTranspose(code_size//2, 3, padding="same", strides=2))
    decoder.add(Conv2DTranspose(code_size//4, 2, padding="same", strides=2))
    decoder.add(Conv2DTranspose(3, 3, padding="same"))
    decoder.add(Conv2D(3, 1, activation='sigmoid'))
    return encoder, decoder


IMG_SHAPE = glomeruli_data.shape[1:]
print(IMG_SHAPE)

encoder, decoder = build_autoencoder_dense(IMG_SHAPE, 1024)

inp = Input(IMG_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = Model(inp, reconstruction)
autoencoder.compile(optimizer='adamax', loss='mse')

# print(autoencoder.summary())

X_train_1 = X_train[:int(X_train.shape[0]*0.5)]
X_train_2 = X_train[int(X_train.shape[0]*0.5):]
X_validation_1 = X_validation[:int(X_test.shape[0]*0.5)]
X_validation_2 = X_validation[int(X_test.shape[0]*0.5):]

history = autoencoder.fit(x=X_train_1, y=X_train_1,
                          epochs=250, validation_data=(X_validation_1, X_validation_1))
history = autoencoder.fit(x=X_train_2, y=X_train_2,
                          epochs=250, validation_data=(X_validation_2, X_validation_2))

autoencoder.save('../../saved_model/autoencoderv5_dense')

predicted = autoencoder.predict(X_test)

decoded_imgs = (predicted * 255).astype(int)

np.save("../../images_autoencoder/decoded_imgs_v5_dense.npy", decoded_imgs)
np.save("../../images_autoencoder/original_imgs_v5_dense.npy", X_test)
