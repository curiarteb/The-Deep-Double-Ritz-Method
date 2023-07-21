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

        # Impose Dirichlet boundary conditions
        x = x*inputs
        x = x*(inputs-1)
        return x

class v_net(tf.keras.Model):

    def __init__(self, depth=1, width=10, activation='sigmoid', **kwargs):
        (super(v_net, self).__init__)(**kwargs)

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

        x=x*inputs
        x=x*(inputs-1)
        return x

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

class dderror_u(tf.keras.Model):

    def __init__(self, error_u_net, **kwargs):
        (super(dderror_u, self).__init__)(**kwargs)

        self.error_u = error_u_net

    def call(self, inputs):
        with tf.GradientTape() as tape2:
            tape2.watch(inputs)
            with tf.GradientTape() as tape1:
                tape1.watch(inputs)
                u = self.error_u(inputs)
            du = tape1.gradient(u, inputs)
        ddu = tape2.gradient(du, inputs)
        return ddu