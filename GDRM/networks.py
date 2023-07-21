import tensorflow as tf

class u_net(tf.keras.Model):

    def __init__(self, depth=1, width=10, activation='sigmoid', **kwargs):
        (super(u_net, self).__init__)(**kwargs)
        self.depth = depth
        self.width = width
        self.activation = activation
        self.layers_list = list()

        for _ in range(self.depth):
            self.layers_list.append(tf.keras.layers.Dense(units=self.width, activation=self.activation, use_bias=True))

        self.layers_list.append(tf.keras.layers.Dense(units=1, activation=None, use_bias=False))

    def call(self, inputs):

        x = inputs
        for layer in self.layers_list:
            x = layer(x)

        # Impose Dirichlet boundary condition (for the moment, u_net(x) = 0 in x=0 and x=1)
        x = x*inputs
        x = x*(inputs-1)
        return x


class Tu_net(tf.keras.Model):

    def __init__(self, u_net, T_operator, **kwargs):
        (super(Tu_net, self).__init__)(**kwargs)

        self.u_net = u_net
        self.T_operator = T_operator

    def call(self, inputs):
        u = self.u_net(inputs)
        Tu = self.T_operator(u)
        return Tu

class error_u(tf.keras.Model):

    def __init__(self, u_net, u_analytic, **kwargs):
        (super(error_u, self).__init__)(**kwargs)

        self.u_net = u_net
        self.u_analytic = u_analytic

    def call(self, inputs):
        u = self.u_net(inputs)
        u_analytic = self.u_analytic(inputs)
        return u-u_analytic

class derror_u(tf.keras.Model):

    def __init__(self, error_u_net, **kwargs):
        (super(derror_u, self).__init__)(**kwargs)

        self.error_u = error_u_net

    def call(self, inputs):
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            u = self.error_u(inputs)
        du = tape.gradient(u, inputs)
        return du

class error_Tu(tf.keras.Model):

    def __init__(self, Tu_net, Tu_analytic, **kwargs):
        (super(error_Tu, self).__init__)(**kwargs)

        self.Tu_net = Tu_net
        self.Tu_analytic = Tu_analytic

    def call(self, inputs):
        Tu = self.Tu_net(inputs)
        Tu_analytic = self.Tu_analytic(inputs)
        return Tu-Tu_analytic

class derror_Tu(tf.keras.Model):

    def __init__(self, error_Tu_net, **kwargs):
        (super(derror_Tu, self).__init__)(**kwargs)

        self.error_Tu = error_Tu_net

    def call(self, inputs):
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            Tu = self.error_Tu(inputs)
        dTu = tape.gradient(Tu, inputs)
        return dTu