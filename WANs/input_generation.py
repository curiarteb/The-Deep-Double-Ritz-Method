import tensorflow as tf
import settings
tf.random.set_seed(1234)
from analytic import f_fn

#For the Beta distribution
import tensorflow_probability as tfp
tfd = tfp.distributions

#For a Beta distribution:
a = tf.constant([1,1], dtype="float64")
b = tf.constant([1,1], dtype="float64")
dist = tfd.Beta(a, b)

def generate_sample(size):
    if settings.is_data_constant:
        x = tf.sort(tf.random.stateless_uniform(shape=[size-2, 1], minval=0., maxval=1., seed=(1, 2)), axis=0)
    else:
        #x = tf.sort(tf.random.uniform(shape=[size-2, 1], minval=10**(-8), maxval=1., seed=1234), axis=0)
        x = tf.reshape(dist.sample(size//2, seed=1234), shape=[size,1])
        x = tf.sort(x, axis=0)

    #x = tf.concat([tf.constant([[0.]]), x, tf.constant([[1.]])], axis=0)
    f = f_fn(x)
    return [x, f]