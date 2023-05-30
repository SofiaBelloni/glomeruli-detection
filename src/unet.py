import tensorflow as tf

class UNet(tf.keras.Model):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder
        self.encoder1 = self.conv_block(64)
        self.encoder2 = self.conv_block(128)
        self.encoder3 = self.conv_block(256)
        self.encoder4 = self.conv_block(512)
        #unet5: self.encoder4 = self.conv_block(1024)

        # Bridge
        self.bridge = self.conv_block(1024)
        #unet5: self.bridge = self.conv_block(2048)

        # Decoder
        #unet: self.decoder5 = self.upconv_block(1024)
        self.decoder4 = self.upconv_block(512)
        self.decoder3 = self.upconv_block(256)
        self.decoder2 = self.upconv_block(128)
        self.decoder1 = self.upconv_block(64)

        # Output
        self.output_conv = layers.Conv2D(1, 1, activation='sigmoid') #il paper usa softmax

    def call(self, inputs):
        # Encoder
        x1 = self.encoder1(inputs)
        x2 = self.encoder2(self.pool(x1))
        x3 = self.encoder3(self.pool(x2))
        x4 = self.encoder4(self.pool(x3))
        #unet5: x5 = self.encoder4(self.pool(x4)) 

        # Bridge
        bridge = self.bridge(self.pool(x4))
        #unet5: bridge = self.bridge(self.pool(x5))

        # Decoder
        #unet5: x = self.decoder5(bridge)
        #       x=self.decoder4(tf.concat([x, x5], axis=-1))
        x = self.decoder4(bridge)
        x = self.decoder3(tf.concat([x, x4], axis=-1))
        x = self.decoder2(tf.concat([x, x3], axis=-1))
        x = self.decoder1(tf.concat([x, x2], axis=-1))

        # Output
        output = self.output_conv(x)

        return output

    def conv_block(self, filters):
        block = tf.keras.Sequential([
            layers.Conv2D(filters, 3, activation='relu', padding='same'),
            layers.Conv2D(filters, 3, activation='relu', padding='same')
        ])
        return block

    def upconv_block(self, filters):
        block = tf.keras.Sequential([
            layers.Conv2DTranspose(filters, 2, strides=2, activation='relu', padding='same')
        ])
        return block

    def pool(self, inputs):
        return layers.MaxPooling2D(2)(inputs)
