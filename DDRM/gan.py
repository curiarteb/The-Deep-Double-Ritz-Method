import tensorflow as tf
tf.keras.backend.set_floatx('float64')
from losses import lossu, lossT
import settings
from input_generation import generate_sample

class GAN(tf.keras.Model):
    def __init__(self, u_net, Tu_net, u_trainable=False, T_trainable=True):
        super(GAN, self).__init__()
        self.u_net = u_net
        self.Tu_net = Tu_net
        self.lossT = lossT(self.u_net, self.Tu_net)
        self.lossu = lossu(self.Tu_net)
        self.u_trainable = tf.Variable(u_trainable)
        self.T_trainable = tf.Variable(T_trainable)

    @tf.function
    def call(self, inputs):
        loss_u = self.lossu(inputs)
        loss_T = self.lossT(inputs)
        return [loss_u, loss_T]

    @tf.function
    def branch_ind(self):
        if self.u_trainable:
            return 0
        elif self.T_trainable:
            return 1
        else:
            return 2

    @tf.function
    def u_train_fn(self, inputs):
        with tf.GradientTape(persistent=True) as tape:
            out_u, out_T = self(inputs)

        grads_u = tape.gradient(out_u, self.u_net.trainable_variables)
        self.u_optimizer.apply_gradients(zip(grads_u, self.u_net.trainable_variables))

        return out_u, out_T

    @tf.function
    def T_train_fn(self, inputs):
        with tf.GradientTape(persistent=True) as tape:
            out_u, out_T = self(inputs)

        grads_T = tape.gradient(out_T, self.Tu_net.T_net.trainable_variables)
        self.T_optimizer.apply_gradients(zip(grads_T, self.Tu_net.T_net.trainable_variables))


        return out_u, out_T


    def compile(self, u_optimizer, T_optimizer):
        super(GAN, self).compile()
        self.u_optimizer = u_optimizer
        self.T_optimizer = T_optimizer


    def train_step(self, data):

        # Sample random points of x
        inputs = generate_sample(settings.integration_sample_size)

        branch_index = self.branch_ind()
        out_u, out_T = tf.switch_case(branch_index, branch_fns=[lambda: self.u_train_fn(inputs), lambda: self.T_train_fn(inputs)])

        return {"loss_u": out_u, "loss_T": out_T}
