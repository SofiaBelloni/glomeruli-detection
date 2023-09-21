import tensorflow as tf
from tune import generate_dataset, tuning


path_to_model = f'../../saved_model/resnet50_.h5'
path_to_feature_extractor = f'../../saved_model/resnet50_feature_extractor_.h5'


def get_resnet():
    return tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(200, 200, 3), pooling="avg", classifier_activation="softmax")


model = get_resnet()
dataset = generate_dataset()
model, feature_extractor = tuning(model, dataset)

model.save(path_to_model)
feature_extractor.save(path_to_feature_extractor)
