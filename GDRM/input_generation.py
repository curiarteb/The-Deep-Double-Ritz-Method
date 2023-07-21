import tensorflow as tf
import settings
tf.random.set_seed(1234)
from analytic import f_fn

#For the Beta distribution
import tensorflow_probability as tfp
tfd = tfp.distributions

#For a Beta distribution:
a = tf.constant([1,10], dtype="float64")
b = tf.constant([1,10], dtype="float64")
dist = tfd.Beta(a, b)

@tf.function
def change_var(x, base=10**20):
    bas=tf.constant(base,dtype=tf.float64)
    return (tf.math.pow(bas, x)-1)/(bas-1)

def generate_sample(size):
    if settings.is_data_constant:
        #x = tf.sort(tf.random.stateless_uniform(shape=[size-2, 1], minval=0., maxval=1., seed=(1, 2)), axis=0)
        x = tf.reshape(dist.sample(size, seed=(1234,1234)), shape=[size,1])
        x = tf.sort(x, axis=0)
    else:
        x = tf.reshape(dist.sample(size//2, seed=1234), shape=[size,1])
        #y = change_var(x)
        #x = tf.concat([tf.reshape(dist.sample(size//2, seed=1234), shape=[size//2,1]), y], axis=0)
        x = tf.sort(x, axis=0)

    #x = tf.concat([tf.constant([[0.]]), x, tf.constant([[1.]])], axis=0)
    f = f_fn(x)
    return [x, f]