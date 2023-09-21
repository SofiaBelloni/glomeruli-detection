import tensorflow as tf
from tune import generate_dataset, tuning

path_to_model = f'../../saved_model/vgg19_.h5'
path_to_feature_extractor = f'../../saved_model/vgg19_feature_extractor_.h5'


def get_vgg19():
    return tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=(200, 200, 3), pooling="avg", classifier_activation="softmax")


model = get_vgg19()
dataset = generate_dataset()
model, feature_extractor = tuning(model, dataset)

model.save(path_to_model)
feature_extractor.save(path_to_feature_extractor)
