import tensorflow as tf
from quadrature_rules import monte_carlo as MC
from quadrature_rules import gaussian_quadrature as gaussian
from forms import b
import settings

class b_norm(tf.keras.layers.Layer):
    def __init__(self, net, with_monte_carlo = settings.monte_carlo, **kwargs):
        (super(b_norm, self).__init__)(**kwargs)
        self.net = net
        self.b = b(self.net, self.net, with_monte_carlo = with_monte_carlo)

    @tf.function
    def call(self, x):
        b_value = self.b(x)
        return tf.sqrt(b_value)

class H1_norm(tf.keras.layers.Layer):
    def __init__(self, net, with_monte_carlo = settings.monte_carlo, **kwargs):
        (super(H1_norm, self).__init__)(**kwargs)
        self.net = net
        if with_monte_carlo:
            self.integrate = MC()
        else:
            self.integrate = gaussian()

    @tf.function
    def call(self, x):
        v = self.net(x)
        dv = tf.gradients(v, x)[0]
        v2 = tf.square(v)
        dv2 = tf.square(dv)
        result = self.integrate(x,v2+dv2)
        return tf.sqrt(result)

class L2_norm(tf.keras.layers.Layer):
    def __init__(self, net, with_monte_carlo = settings.monte_carlo, **kwargs):
        (super(L2_norm, self).__init__)(**kwargs)
        self.net = net
        if with_monte_carlo:
            self.integrate = MC()
        else:
            self.integrate = gaussian()

    @tf.function
    def call(self, x):
        v = self.net(x)
        v2 = tf.square(v)
        result = self.integrate(x,v2)
        return tf.sqrt(result)