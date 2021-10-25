import tensorflow as tf
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
            h = tf.keras.layers.Dense(units=num_neurons, activation='sigmoid')(h)
            disaggregation_outputs.append(h)
        self.disaggregation_model = tf.keras.Model(inputs=disaggregator_input, outputs=disaggregation_outputs)
        self.disaggregation_model.summary()

        # define the decoder network
        decoder_input = tf.keras.Input(int(config['disaggregation_layers_fracs'][-1]*self.data_interface.num_classes))
        h = decoder_input
        for frac in reversed(config['disaggregation_layers_fracs'][:-1]):
            num_neurons = int(frac * self.data_interface.num_classes)
            h = tf.keras.layers.Dense(units=num_neurons, activation='relu')(h)
        h = tf.keras.layers.Dense(units=self.data_interface.num_classes, activation='relu')(h)
        self.decoder_model = tf.keras.Model(inputs=decoder_input, outputs=h)
        self.decoder_model.summary()

        #define the additional loss terms
        self.decoder_loss = tf.keras.metrics.Mean(name='decoder_loss')
        self.disaggregation_loss = tf.keras.metrics.Mean(name='disaggregation_loss')

    # the training step for learning the hard way
    @tf.function
    def train_step(self, x, y):
        pass