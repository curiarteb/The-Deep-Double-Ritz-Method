import tensorflow as tf
tf.keras.backend.set_floatx('float64')
tf.random.set_seed(1234)
from forms import b, l
from norms import b_norm

class lossuv(tf.keras.layers.Layer):
    def __init__(self, u_net, v_net, **kwargs):
        (super(lossuv, self).__init__)(**kwargs)
        self.u_net = u_net
        self.v_net = v_net
        self.b = b(self.u_net, self.v_net)
        self.l = l(self.v_net)
        self.norm = b_norm(self.v_net)

    @tf.function
    def call(self, inputs):
        x,f = inputs
        b_out = self.b(x)
        l_out = self.l(inputs)
        norm = self.norm(x)
        result = tf.math.abs(b_out - l_out)
        result = result / norm
        #tf.print("out", result)
        return result