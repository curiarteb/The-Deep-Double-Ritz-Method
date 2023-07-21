import tensorflow as tf
tf.keras.backend.set_floatx('float64')
tf.random.set_seed(1234)
from losses import lossu
import settings
from input_generation import generate_sample

class GAN(tf.keras.Model):
    def __init__(self, Tu_net, u_trainable=True):
        super(GAN, self).__init__()
        self.Tu_net = Tu_net
        self.lossu = lossu(self.Tu_net)
        self.u_trainable = tf.Variable(u_trainable)

    @tf.function
    def call(self, inputs):
        loss_u = self.lossu(inputs)
        return loss_u

    @tf.function
    def u_train_fn(self, inputs):
        with tf.GradientTape(persistent=True) as tape:
            out_u = self(inputs)

        grads_u = tape.gradient(out_u, self.Tu_net.trainable_variables)
        self.u_optimizer.apply_gradients(zip(grads_u, self.Tu_net.trainable_variables))

        grads_concat = tf.concat([tf.reshape(tf.abs(g), shape=[-1]) for g in grads_u], 0)
        grads_abs_mean = tf.reduce_mean(grads_concat)
        grads_abs_max = tf.reduce_max(grads_concat)
        grads_abs_min = tf.reduce_min(grads_concat)

        return out_u, grads_abs_max, grads_abs_mean, grads_abs_min

    def compile(self, u_optimizer):
        super(GAN, self).compile()
        self.u_optimizer = u_optimizer


    def train_step(self, data):

        # Sample random points of x
        inputs = generate_sample(settings.integration_sample_size)

        out_u = self.u_train_fn(inputs)

        return {"loss_u": out_u}
