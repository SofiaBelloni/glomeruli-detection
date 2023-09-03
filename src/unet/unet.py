import tensorflow as tf


def upsample(filters, size):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))
  result.add(tf.keras.layers.ReLU())

  return result

def Unet(output_channels=2, input_shape=[512, 512, 3]):
  base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False)
  layer_names = [
    'block_1_expand_relu',   # 256x256
    'block_3_expand_relu',   # 128x128
    'block_6_expand_relu',   # 64x64
    'block_13_expand_relu',  # 32x32
    'block_16_project',      # 16x16
  ]
  base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

  # Create the feature extraction model
  down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
  down_stack.trainable = False

  up_stack = [
    upsample(512, 3),  # 16x16 -> 32x32
    upsample(256, 3),  # 32x32 -> 64x64
    upsample(128, 3),  # 64x64 -> 128x128
    upsample(64, 3),   # 128x128 -> 256x256
  ]

  inputs = tf.keras.layers.Input(shape=input_shape)

  # Downsampling through the model
  skips = down_stack(inputs)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      filters=output_channels, kernel_size=3, strides=2,
      padding='same')  #256x256 -> 512x512

  x = last(x)
  x = tf.keras.layers.Softmax(axis=-1)(x)

  return tf.keras.Model(inputs=inputs, outputs=x)