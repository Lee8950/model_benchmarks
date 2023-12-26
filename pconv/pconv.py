import tensorflow as tf

class PConv2D(tf.keras.layers.Layer):
    def __init__(self, dim, n_div):
        super().__init__()
        self.p_dim = dim // n_div
        self.untouched = dim - self.p_dim
        self.conv2d = tf.keras.layers.Conv2D(self.p_dim,
                                            kernel_size=(3,3),
                                            strides=(1,1),
                                            padding="same",
                                            data_format="channels_last",
                                            )

    def call(self, inputs):
        x1, x2 = tf.split(inputs, [self.p_dim, self.untouched], axis=-1)
        x1 = self.conv2d(x1)
        result = tf.concat([x1, x2], -1)
        return result

class MLPBlock(tf.keras.layers.Layer):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.p_dim = dim
        self.first = tf.keras.layers.Conv2D(self.hidden_dim,
                                            kernel_size=(1,1),
                                            strides=(1,1),
                                            padding="same",
                                            data_format="channels_last",
                                            )
        self.activation = tf.keras.layers.ReLU()
        self.second = tf.keras.layers.Conv2D(self.p_dim,
                                            kernel_size=(1,1),
                                            strides=(1,1),
                                            padding="same",
                                            data_format="channels_last",
                                            )

    def call(self, inputs):
        return self.second(self.activation(self.first(inputs)))