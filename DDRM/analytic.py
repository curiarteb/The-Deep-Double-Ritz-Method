import tensorflow as tf
tf.keras.backend.set_floatx('float64')

#For the Beta distribution:
import tensorflow_probability as tfp
tfd = tfp.distributions

# Polynomial
alpha=0.8
alpha = tf.constant(alpha, dtype=tf.float64)
@tf.function
def f_fn(x):
    return -(alpha + 1) * alpha * tf.pow(x, alpha-1) + alpha * (alpha - 1) * tf.pow(x, alpha - 2)
    #return -2*tf.ones_like(x)

@tf.function
def u_analytic_fn(x):
    return tf.pow(x,alpha)*(x-1)

@tf.function
def Tu_analytic_fn(x):
    return tf.pow(x,alpha)*(x-1)

size = 10**4

x = tf.range(start=0.+1/(size), limit=1., delta=1./(size), dtype=tf.float64)
x = tf.reshape(x,shape=[x.shape[0],1])
optimal_loss_u = alpha/(2*(1-4*tf.square(alpha)))
optimal_loss_T = alpha/(2*(1-4*tf.square(alpha)))
optimal_u_norm = tf.sqrt(alpha/(-1+4*tf.square(alpha)))
optimal_Tu_norm = tf.sqrt(alpha/(-1+4*tf.square(alpha)))
