import tensorflow as tf
tf.keras.backend.set_floatx('float64')
tf.random.set_seed(1234)
from forms import b, l
from norms import b_norm

class lossT(tf.keras.layers.Layer):
    def __init__(self, u_net, Tu_net, **kwargs):
        (super(lossT, self).__init__)(**kwargs)
        self.u_net = u_net
        self.Tu_net = Tu_net
        self.b = b(self.u_net, self.Tu_net)
        self.norm = b_norm(self.Tu_net)

    @tf.function
    def call(self, inputs):
        x,f = inputs
        b_out = self.b(x)
        norm = self.norm(x)
        result = 1/2*tf.square(norm) - b_out
        return result

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
        norm_out = self.norm(x)
        result = 1/2*tf.square(norm_out) - l_out
        return result