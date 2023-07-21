import tensorflow as tf
tf.keras.backend.set_floatx('float64')
tf.random.set_seed(1234)
from forms import b, l
from norms import H1_norm, b_norm, L2_norm

class lossu(tf.keras.layers.Layer):
    def __init__(self, Tu_net, **kwargs):
        (super(lossu, self).__init__)(**kwargs)
        self.Tu_net = Tu_net
        self.l = l(self.Tu_net)
        self.norm = b_norm(self.Tu_net)

    @tf.function
    def call(self, inputs):
        x,f = inputs
        l_out = self.l(inputs)
        norm = self.norm(x)
        result = 1/2*tf.square(norm) - l_out
        return result