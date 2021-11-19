import tensorflow as tf

print(tf.config.list_physical_devices("GPU"))

layer = tf.keras.layers.Dense(100)
layer = tf.keras.layers.Dense(10, input_shape=(None, 5))

layer(tf.zeros([10, 5]))

layer.variables

layer.kernel
layer.bias

class MyDenseLayer(tf.keras.layers.layer):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_weight
