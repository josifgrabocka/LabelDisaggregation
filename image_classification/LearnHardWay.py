import tensorflow as tf
import tensorflow_addons as tfa
from image_classification.DefaultOptimizer import DefaultOptimizer
import time
import datetime


class LearnHardWay(DefaultOptimizer):

    def __init__(self, prediction_model, data_interface, config):
        super(LearnHardWay, self).__init__(prediction_model=prediction_model, data_interface=data_interface, config=config)

        # define the label disaggregation model
        disaggregator_input = tf.keras.Input(self.data_interface.num_classes)
        h = disaggregator_input
        for idx, frac in enumerate(config['disaggregation_layers_fracs']):
            units=int(frac * self.data_interface.num_classes)
            if units == 0:
                units = 1
            # add a dense layer with relu activations, unless it is the last layer where the activation is None
            h = tf.keras.layers.Dense(units=units, activation=None)(h)
            if idx < len(config['disaggregation_layers_fracs'])-1:
                h = tf.keras.layers.Activation('selu')(h)

        self.disaggregation_model = tf.keras.Model(inputs=disaggregator_input, outputs=h)
        self.disaggregation_model.summary()

        # define the label disaggregation approximation model
        disaggregator_input = tf.keras.Input(self.data_interface.num_classes)
        h = disaggregator_input
        for idx, frac in enumerate(config['disaggregation_layers_fracs']):
            units = int(frac * self.data_interface.num_classes)
            if units == 0:
                units = 1
            # add a dense layer with relu activations, unless it is the last layer where the activation is None
            h = tf.keras.layers.Dense(units=units, activation=None)(h)
            if idx < len(config['disaggregation_layers_fracs']) - 1:
                h = tf.keras.layers.Activation('selu')(h)

        self.disaggregation_approx_model = tf.keras.Model(inputs=disaggregator_input, outputs=h)
        self.disaggregation_approx_model.summary()

        #define the additional loss terms metrics
        self.disaggregation_loss_metric = tf.keras.metrics.Mean(name='disaggregation_loss')
        self.logs_metrics.append(self.disaggregation_loss_metric)
        # add the additional loss term definitions
        self.disaggregation_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        # the cosine decay learning rate scheduler with restarts and the decoupled L2 adam with gradient clipping
        step = tf.Variable(0, trainable=False)
        lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=config['eta'],
                                                                     t_mul=1,
                                                                     first_decay_steps=self.first_decay_steps)
        wd = self.l2_penalty * lr_sched(step)
        self.disaggregation_optimizer = tfa.optimizers.AdamW(learning_rate=lr_sched, weight_decay=wd)

        # the cosine decay learning rate scheduler with restarts and the decoupled L2 adam with gradient clipping
        step = tf.Variable(0, trainable=False)
        lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=config['eta'],
                                                                     t_mul=1,
                                                                     first_decay_steps=self.first_decay_steps)
        wd = self.l2_penalty * lr_sched(step)
        self.disaggregation_approx_optimizer = tfa.optimizers.AdamW(learning_rate=lr_sched, weight_decay=wd)

        # populate the list of models added by child classes of the DefaultOptimizer, such that the saving of these models
        # can happen inside the run method of the parent class
        self.child_classes_models.append(self.disaggregation_model)
        self.disaggregation_model_file_prefix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '_disaggregation_model_' \
                                            + self.config['model_name'] + '_' + self.config['dataset_name'] + '_' + self.config['learning_style']
        self.child_classes_model_file_prefixes.append(self.disaggregation_model_file_prefix)

        # the hyper-parameter for the strength of
        self.disaggregation_gamma = self.config['gamma']

    # the training step for learning the hard way
    @tf.function
    def train_step(self, x, y):


        with tf.GradientTape(persistent=True) as tape:

            y_pred = self.prediction_model(x, training=True)
            loss_y = self.cat_loss(y_true=y, y_pred=y_pred)

            # define the loss of the disaggregation
            z_true = self.disaggregation_model(y, training=True)
            z_pred = self.disaggregation_approx_model(y_pred, training=True)
            z_true_one_hot = tf.one_hot(tf.argmax(z_true, axis=1), depth=int(self.config['disaggregation_layers_fracs'][-1] * self.data_interface.num_classes))
            #z_pred_one_hot = tf.one_hot(tf.argmax(z_pred, axis=1), depth=int(self.config['disaggregation_layers_fracs'][-1] * self.data_interface.num_classes))

            loss_z = self.disaggregation_loss(y_true=z_pred, y_pred=z_true)
            loss_z_approx = self.disaggregation_loss(y_true=z_true_one_hot, y_pred=z_pred)

            if self.config['lhw_mode'] == 'lhw':
                loss_prediction_model = loss_y + loss_z_approx
                loss_disaggregation_model = -tf.tanh(self.disaggregation_gamma*loss_z)
                loss_disaggregation_approx_model = -tf.tanh(self.disaggregation_gamma * loss_z_approx)
            elif self.config['lhw_mode'] == 'random':
                loss_prediction_model = loss_y + loss_z_approx
            elif self.config['lhw_mode'] == 'max':
                loss_disaggregation_model = -tf.tanh(self.disaggregation_gamma*loss_z)
                loss_disaggregation_approx_model = -tf.tanh(self.disaggregation_gamma * loss_z_approx)

        # update the prediction model params
        if self.config['lhw_mode'] == 'lhw' or self.config['lhw_mode'] == 'random':
            prediction_model_weights = self.prediction_model.trainable_variables
            prediction_gradients = tape.gradient(loss_prediction_model, prediction_model_weights)
            self.prediction_optimizer.apply_gradients(zip(prediction_gradients, prediction_model_weights))

        # update the disaggregation model
        if self.config['lhw_mode'] == 'lhw' or self.config['lhw_mode'] == 'max':
            disaggregation_model_weights = self.disaggregation_model.trainable_variables
            disaggregation_gradients = tape.gradient(loss_disaggregation_model, disaggregation_model_weights)
            self.disaggregation_optimizer.apply_gradients(zip(disaggregation_gradients, disaggregation_model_weights))

            disaggregation_approx_model_weights = self.disaggregation_approx_model.trainable_variables
            disaggregation_approx_gradients = tape.gradient(loss_disaggregation_approx_model, disaggregation_approx_model_weights)
            self.disaggregation_approx_optimizer.apply_gradients(zip(disaggregation_approx_gradients, disaggregation_approx_model_weights))

        # update the metrics
        self.train_loss(loss_y)
        self.train_accuracy(y, y_pred)
        self.disaggregation_loss_metric(loss_z_approx)

    # def run(self):
    #
    #     print('Pretrain the disaggregator with a toy model ...')
    #
    #     model = self.config['model_name']
    #     num_epochs = self.config['num_epochs']
    #     lew_mode = self.config['lew_mode']
    #     eta = self.config['eta']
    #
    #     # set some mini configurations
    #     self.config['model_name'] = 'mini'
    #     self.config['lhw_mode'] = 'lew'
    #     self.config['num_epochs'] = 100
    #     self.config['eta'] = 0.001
    #
    #     super().run()
    #
    #     print('Now learning the real model ...')
    #
    #     self.config['model_name'] = model
    #     self.config['lew_mode'] = lew_mode
    #     self.config['num_epochs'] = num_epochs
    #     self.config['eta'] = eta
    #
    #     # reinit the optimizers
    #     step = tf.Variable(0, trainable=False)
    #     lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=self.config['eta'], t_mul=1.0, first_decay_steps=self.first_decay_steps)
    #     wd = self.l2_penalty * lr_sched(step)
    #     self.prediction_optimizer = tfa.optimizers.AdamW(learning_rate=lr_sched, weight_decay=wd)
    #
    #     step = tf.Variable(0, trainable=False)
    #     lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=self.config['eta'], t_mul=1, first_decay_steps=self.first_decay_steps)
    #     wd = self.l2_penalty * lr_sched(step)
    #     self.disaggregation_optimizer = tfa.optimizers.AdamW(learning_rate=lr_sched, weight_decay=wd)
    #
    #     super().run()
