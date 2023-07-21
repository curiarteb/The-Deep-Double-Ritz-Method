import tensorflow as tf
from quadrature_rules import monte_carlo as MC
from quadrature_rules import gaussian_quadrature as gaussian
import settings

class b(tf.keras.layers.Layer):
    def __init__(self, net1, net2, with_monte_carlo = settings.monte_carlo, **kwargs):
        (super(b, self).__init__)(**kwargs)
        self.net1 = net1
        self.net2 = net2
        if with_monte_carlo:
            self.integrate = MC()
        else:
            self.integrate = gaussian()

    # General bilinear form for -u''=f
    @tf.function
    def call(self, x):
        out1 = self.net1(x)
        out2 = self.net2(x)
        dout1 = tf.gradients(out1, x)[0]
        dout2 = tf.gradients(out2, x)[0]
        #integrand = dout1 * dout2
        integrand = tf.einsum("ki,ki->ki", dout1, dout2)
        result = self.integrate(x, integrand)
        return result

class l(tf.keras.layers.Layer):
    def __init__(self, net, with_monte_carlo = settings.monte_carlo, **kwargs):
        (super(l, self).__init__)(**kwargs)
        self.net = net
        if with_monte_carlo:
            self.integrate = MC()
        else:
            self.integrate = gaussian()

    # General linear form for -u''=f
    @tf.function
    def call(self, inputs):
        x,f = inputs
        out = self.net(x)
        integrand = f*out
        result = self.integrate(x, integrand)
        return result
