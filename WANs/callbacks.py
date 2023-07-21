import tensorflow as tf
import numpy as np
import math
tf.keras.backend.set_floatx('float64')
tf.random.set_seed(1234)
import settings

class alternate_training(tf.keras.callbacks.Callback):
    def __init__(self):
        #self.patience_v = 3
        #self.patience_u = 100
        self.u_counter = 0
        self.v_counter = 0
        #self.wait_u = 0
        #self.wait_v = 0
        self.u_max_iterations = settings.u_max_iterations
        self.v_max_iterations = settings.v_max_iterations

    def on_train_begin(self, logs={}):
        print("Training begins.\n")
        self.stopped_epoch = 0
        self.best_loss_u = np.Inf
        self.best_loss_T = np.Inf
        self.best_weights = None


    def on_epoch_begin(self, epoch, logs={}):
        #tf.print("v variables", self.model.v_net.variables)
        #tf.print("v optimizer outside begin", self.model.v_optimizer.variables())
        #if self.model.u_trainable:
        #    print("u trainable", self.u_counter)
        #if self.model.v_trainable:
        #    print("v trainable", self.v_counter)

        if self.model.u_trainable:
            self.u_counter += 1
        if self.model.v_trainable:
            self.v_counter += 1

    def on_epoch_end(self, epoch, logs={}):

        #loss_u = logs["loss_u"]
        #loss_T = logs["loss_T"]

        if self.model.u_trainable:
            if self.u_counter >= self.u_max_iterations:
                self.model.u_trainable.assign(False)
                self.model.v_trainable.assign(True)
                self.u_counter = 0
                self.v_counter = 0
                #self.wait_u += 1

        if self.model.v_trainable:
            if self.v_counter >= self.v_max_iterations:
                self.model.u_trainable.assign(True)
                self.model.v_trainable.assign(False)
                self.u_counter = 0
                self.v_counter = 0
                #self.wait_T += 1
                #for var in self.model.T_optimizer.variables():
                #    var.assign(tf.zeros_like(var))

        # if loss_T < self.best_loss_T:
        #     self.best_loss_T = loss_T
        #     self.wait_T = 0
        #     self.T_max_iterations = settings.T_max_iterations
        #
        # if loss_u < self.best_loss_u:
        #     self.best_loss_u = loss_u
        #     self.best_weights = self.model.get_weights()
        #     self.wait_u = 0
        #
        # if self.wait_T >= self.patience_T:
        #     self.T_max_iterations += 5
        #     self.wait_u = 0
        #
        # if self.wait_u >= self.patience_u:
        #     self.stopped_epoch = epoch
        #     self.model.stop_training = True
        #     print("Restoring model weights from the end of the best epoch.")
        #     self.model.set_weights(self.best_weights)




