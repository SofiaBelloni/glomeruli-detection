import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from elaborate_data import data_aug_impl_no_label, glomeruli_crop_dark_backgroung_super_cool, glomeruli_crop, data_aug_impl
from model import build_cnn_dense_autoencoder

latent_space_dim = 512
version = 2.0
batch_size = 4
epochs = 500


def load_data():
    glomeruli_data = np.load(
        "../../dataset512x512/glomeruli_data_no_dataaug.npy")
    X_train, X_test = train_test_split(
        glomeruli_data, test_size=0.1, random_state=42)
    return X_train, X_test


def generate_data():
    glomeruli_data = np.load(
        "../../dataset512x512/glomeruli_data_no_dataaug.npy")
    X_train, X_vt = train_test_split(
        glomeruli_data, test_size=0.5, random_state=42)
    X_validation, X_test = train_test_split(
        X_vt, test_size=0.5, random_state=42)
    X_train = (X_train * 255).astype(np.uint8)
    X_train = data_aug_impl_no_label(X_train, 3)
    X_train = X_train.astype('float32') / 255
    return X_train, X_test, X_validation


def generate_dark_data():
    dataset = np.load(
        "../../dataset512x512/dataset.npy")
    labels = np.load(
        "../../dataset512x512/labels.npy")

    glomeruli_data, glomeruli_labels, _, _ = glomeruli_crop(dataset, labels)

    X_train, X_vt = train_test_split(
        glomeruli_data, test_size=0.5, random_state=42)
    X_labels, X_vt_labels = train_test_split(
        glomeruli_labels, test_size=0.5, random_state=42)

    X_validation, X_test = train_test_split(
        X_vt, test_size=0.5, random_state=42)
    X_validation_l, X_test_l = train_test_split(
        X_vt_labels, test_size=0.5, random_state=42)
    # X_train = (X_train * 255).astype(np.uint8)
    X_train, X_labels = data_aug_impl(X_train[0].shape, X_train, X_labels)

    X_train_label, _ = glomeruli_crop_dark_backgroung_super_cool(
        X_train, X_labels)
    X_validation_labels, _ = glomeruli_crop_dark_backgroung_super_cool(
        X_validation, X_validation_l)
    X_test_labels, _ = glomeruli_crop_dark_backgroung_super_cool(
        X_test, X_test_l)

    X_train = X_train.astype('float32') / 255
    X_train_label = X_train_label.astype('float32') / 255
    X_validation = X_validation.astype('float32') / 255
    X_validation_labels = X_validation_labels.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    X_test_labels = X_test_labels.astype('float32') / 255
    return X_train, X_validation, X_test, X_train_label, X_validation_labels, X_test_labels


def build_and_compile_model(img_shape, code_size):
    encoder, decoder = build_cnn_dense_autoencoder(img_shape, code_size)

    inp = Input(img_shape)
    code = encoder(inp)
    reconstruction = decoder(code)

    autoencoder = tf.keras.Model(inputs=inp, outputs=reconstruction)
    autoencoder.compile(optimizer='adamax', loss='mse')
    return autoencoder


def train_model(model, X_train, epochs, batch_size, X_validation=None, X_validation_label=None, X_train_label=None):
    early_stopping = EarlyStopping(
        monitor='val_loss', start_from_epoch=75, patience=20, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(
        f'../../saved_model/autoencoder_v{version}_dense_{latent_space_dim}.h5', save_best_only=True, monitor='val_loss', mode='min')
    if X_validation is None:
        if X_train_label is not None:
            history = model.fit(
                X_train, X_train_label,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=[early_stopping, model_checkpoint]
            )
            return model, history
        else:
            history = model.fit(
                X_train, X_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=[early_stopping, model_checkpoint]
            )
            return model, history
    else:
        if X_train_label is not None:
            history = model.fit(
                X_train, X_train_label,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_validation, X_validation_label),
                callbacks=[early_stopping, model_checkpoint]
            )
            return model, history
        else:
            history = model.fit(
                X_train, X_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_validation, X_validation),
                callbacks=[early_stopping, model_checkpoint]
            )
            return model, history


def save_decoded_images(model, X_test):
    predicted = model.predict(X_test)
    decoded_imgs = (predicted * 255).astype(int)
    np.save(
        f"../../images_autoencoder/decoded_imgs_v{version}_dense_{latent_space_dim}.npy", decoded_imgs)
    np.save(
        f"../../images_autoencoder/original_imgs_v{version}_dense_{latent_space_dim}.npy", X_test)


def main():
    # X_train, X_test = load_data()
    # X_train, X_test, X_validation = generate_data()

    X_train, X_validation, X_test, X_train_dark, X_validation_dark, X_test_dark = generate_dark_data()

    IMG_SHAPE = X_train.shape[1:]
    print(IMG_SHAPE)

    autoencoder = build_and_compile_model(IMG_SHAPE, latent_space_dim)
    autoencoder, history = train_model(
        autoencoder, X_train, epochs=epochs, batch_size=batch_size, X_validation=X_validation, X_validation_label=X_validation_dark, X_train_label=X_train_dark)
    # autoencoder, history = train_model(
    #     autoencoder, X_train_dark, epochs=epochs, batch_size=batch_size, X_validation=X_validation_dark)

    # autoencoder.save(f'../../saved_model/autoencoderv{version}_dense_{latent_space_dim}')
    save_decoded_images(autoencoder, X_test_dark)


if __name__ == "__main__":
    main()
