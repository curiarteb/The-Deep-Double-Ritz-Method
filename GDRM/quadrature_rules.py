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
