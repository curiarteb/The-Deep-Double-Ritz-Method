import tensorflow as tf
tf.keras.backend.set_floatx('float64')
tf.random.set_seed(1234)
from losses import lossuv
import settings
from input_generation import generate_sample

class GAN(tf.keras.Model):
    def __init__(self, u_net, v_net, u_trainable=False, v_trainable=True):
        super(GAN, self).__init__()
        self.u_net = u_net
        self.v_net = v_net
        self.lossuv = lossuv(self.u_net, self.v_net)
        self.u_trainable = tf.Variable(u_trainable)
        self.v_trainable = tf.Variable(v_trainable)

    @tf.function
    def call(self, inputs):
        loss_uv = self.lossuv(inputs)
        return loss_uv

    @tf.function
    def branch_ind(self):
        if self.u_trainable:
            return 0
        elif self.v_trainable:
            return 1
        else:
            return 2

    @tf.function
    def u_train_fn(self, inputs):
        with tf.GradientTape(persistent=True) as tape:
            out_uv = self(inputs)

        grads_u = tape.gradient(out_uv, self.u_net.trainable_variables)
        self.u_optimizer.apply_gradients(zip(grads_u, self.u_net.trainable_variables))


        return out_uv

    @tf.function
    def v_train_fn(self, inputs):
        with tf.GradientTape(persistent=True) as tape:
            out_uv = self(inputs)

        grads_v = tape.gradient(out_uv, self.v_net.trainable_variables)
        grads_v = [-1*g for g in grads_v] #We multiply the gradients by -1 for gradient ascent
        self.v_optimizer.apply_gradients(zip(grads_v, self.v_net.trainable_variables))


        return out_uv


    def compile(self, u_optimizer, v_optimizer):
        super(GAN, self).compile()
        self.u_optimizer = u_optimizer
        self.v_optimizer = v_optimizer


    def train_step(self, data):

        # Sample random points of x
        inputs = generate_sample(settings.integration_sample_size)

        branch_index = self.branch_ind()
        out_uv = tf.switch_case(branch_index, branch_fns=[lambda: self.u_train_fn(inputs), lambda: self.v_train_fn(inputs)])


        return {"loss_uv": out_uv}
