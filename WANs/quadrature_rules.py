import tensorflow as tf
tf.keras.backend.set_floatx('float64')

class monte_carlo(tf.keras.layers.Layer):
    def __init__(self, extreme_points=[0,1], **kwargs):
        (super(monte_carlo, self).__init__)(**kwargs)
        self.extreme_points = extreme_points

    @tf.function
    def call(self, x, f):
        volume=self.extreme_points[1]-self.extreme_points[0]
        return volume*tf.reduce_mean(f)

class trapezoidal_rule(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        (super(trapezoidal_rule, self).__init__)(**kwargs)

    @tf.function
    def call(self, x, f):
        x_left = x[:-1,:]
        x_right = x[1:,:]
        volumes = x_right - x_left
        #tf.print("volumes", volumes)
        f_left = f[1:,:]
        f_right = f[:-1,:]
        f_sums = 1/2*(f_left + f_right)
        #tf.print("images", f_sums)

        return tf.reduce_sum(volumes*f_sums)

class gaussian_quadrature(tf.keras.layers.Layer):
    def __init__(self, extreme_points=[0,1], **kwargs):
        (super(gaussian_quadrature, self).__init__)(**kwargs)
        self.extreme_points = extreme_points

    @tf.function
    def call(self, x, f):
        x_left = x[:-1,:]
        x_right = x[1:,:]
        mid_points = (x_right + x_left)/2
        mid_points_left = tf.concat([tf.constant(self.extreme_points[0], shape=(1,1), dtype=tf.float64), mid_points], axis=0)
        mid_points_right = tf.concat([mid_points, tf.constant(self.extreme_points[1], shape=(1,1), dtype=tf.float64)], axis=0)
        volumes = mid_points_right - mid_points_left

        return tf.reduce_sum(volumes*f)

# a=3
# @tf.function
# def f_fn(x):
#     #return -(a + 1) * a * tf.pow(x, a-1) + a * (a - 1) * tf.pow(x, a - 2)
#     #return -2*tf.ones_like(x)
#     #return -6*x+2
#     return -12*tf.square(x)+6*x
#
# x = tf.sort(tf.random.uniform(shape=[100,1], minval=0., maxval=1., seed=1), axis=0)
# f = f_fn(x)
#
# MC = monte_carlo(volume=1)
# trap = trapezoidal_rule()
#
# print("MC", MC(f))
# print("trap", trap(x,f))