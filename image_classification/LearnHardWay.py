import tensorflow as tf
import tensorflow_addons as tfa
from image_classification.DefaultOptimizer import DefaultOptimizer
import time


class LearnHardWay(DefaultOptimizer):

    def __init__(self, prediction_model, data_interface, config):
        super(LearnHardWay, self).__init__(prediction_model=prediction_model, data_interface=data_interface, config=config)

        # define the label disaggregation model
        disaggregator_input = tf.keras.Input(self.data_interface.num_classes)
        h = disaggregator_input
        disaggregation_outputs = []
        for frac in config['disaggregation_layers_fracs']:
            num_neurons = int(frac*self.data_interface.num_classes)
            h = tf.keras.layers.Dense(units=num_neurons, activation=None)(h)
            h = tf.keras.layers.BatchNormalization()(h)
            h = tf.keras.layers.Activation('sigmoid')(h)

            disaggregation_outputs.append(h)
        self.disaggregation_model = tf.keras.Model(inputs=disaggregator_input, outputs=disaggregation_outputs)
        self.disaggregation_model.summary()

        # define the decoder network
        # decoder_input = tf.keras.Input(int(config['disaggregation_layers_fracs'][-1]*self.data_interface.num_classes))
        # h = decoder_input
        # for frac in reversed(config['disaggregation_layers_fracs'][:-1]):
        #     num_neurons = int(frac * self.data_interface.num_classes)
        #     h = tf.keras.layers.Dense(units=num_neurons, activation='relu')(h)
        # h = tf.keras.layers.Dense(units=self.data_interface.num_classes, activation='relu')(h)
        # self.decoder_model = tf.keras.Model(inputs=decoder_input, outputs=h)
        # self.decoder_model.summary()

        #define the additional loss terms metrics
        self.disaggregation_loss = tf.keras.metrics.Mean(name='disaggregation_loss')
        self.logs_metrics.append(self.disaggregation_loss)
        # add the additional loss term definitions
        self.bin_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        # the cosine decay learning rate scheduler with restarts and the decoupled L2 adam with gradient clipping
        step = tf.Variable(0, trainable=False)
        lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=config['eta'],
                                                                     first_decay_steps=self.first_decay_steps)
        wd = self.l2_penalty * lr_sched(step)
        self.disaggregation_optimizer = tfa.optimizers.AdamW(learning_rate=lr_sched, weight_decay=wd)

    # the training step for learning the hard way
    @tf.function
    def train_step(self, x, y):

        with tf.GradientTape(persistent=True) as tape:
            y_pred = self.prediction_model(x, training=True)
            loss_y = self.cat_loss(y_true=y, y_pred=y_pred)

            # define the loss of the disaggregation
            z_list = self.disaggregation_model(y, training=True)
            z_pred_list = self.disaggregation_model(y_pred, training=True)
            loss_z = tf.reduce_mean([self.bin_loss(y_true=z, y_pred=z_pred) for z, z_pred in zip(z_list, z_pred_list)])

            loss_prediction_model = loss_y + loss_z
            loss_disaggregation_model = -loss_z

        # update the prediction model
        prediction_model_weights = self.prediction_model.trainable_variables
        prediction_gradients = tape.gradient(loss_prediction_model, prediction_model_weights)
        self.prediction_optimizer.apply_gradients(zip(prediction_gradients, prediction_model_weights))

        # update the disaggregation model
        disaggregation_model_weights = self.disaggregation_model.trainable_variables
        disaggregation_gradients = tape.gradient(loss_disaggregation_model, disaggregation_model_weights)
        self.disaggregation_optimizer.apply_gradients(zip(disaggregation_gradients, disaggregation_model_weights))

        # update the metrics
        self.train_loss(loss_y)
        self.train_accuracy(y, y_pred)
        self.disaggregation_loss(loss_z)
