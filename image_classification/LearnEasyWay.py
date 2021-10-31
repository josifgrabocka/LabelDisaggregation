import tensorflow as tf
import tensorflow_addons as tfa
from image_classification.DefaultOptimizer import DefaultOptimizer
import time
import datetime


class LearnEasyWay(DefaultOptimizer):

    def __init__(self, prediction_model, data_interface, config):
        super(LearnEasyWay, self).__init__(prediction_model=prediction_model, data_interface=data_interface, config=config)

        # define the label disaggregation model
        disaggregator_input = tf.keras.Input(self.data_interface.num_classes)
        h = disaggregator_input
        disaggregation_outputs = []
        self.num_disaggregator_units = []
        for frac in config['disaggregation_layers_fracs']:
            units=int(frac * self.data_interface.num_classes)
            if units == 0:
                units = 1
            h = tf.keras.layers.Dense(units=units, activation='softmax')(h)
            disaggregation_outputs.append(h)
            self.num_disaggregator_units.append(units)
        self.disaggregation_model = tf.keras.Model(inputs=disaggregator_input, outputs=disaggregation_outputs)
        self.disaggregation_model.summary()

        #define the additional loss terms metrics
        self.disaggregation_loss_metric = tf.keras.metrics.Mean(name='disaggregation_loss')
        self.logs_metrics.append(self.disaggregation_loss_metric)
        # add the additional loss term definitions
        self.disaggregation_loss = tf.keras.losses.CategoricalCrossentropy()

        # the cosine decay learning rate scheduler with restarts and the decoupled L2 adam with gradient clipping
        step = tf.Variable(0, trainable=False)
        lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=config['eta'],
                                                                     t_mul=1,
                                                                     first_decay_steps=self.first_decay_steps)
        wd = self.l2_penalty * lr_sched(step)
        self.disaggregation_optimizer = tfa.optimizers.AdamW(learning_rate=lr_sched, weight_decay=wd)

        self.dis_loss_step = tf.Variable(0, trainable=False)
        self.dis_loss_lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=config['eta'],
                                                                     t_mul=1,
                                                                     first_decay_steps=self.first_decay_steps)

        # populate the list of models added by child classes of the DefaultOptimizer, such that the saving of these models
        # can happen inside the run method of the parent class
        self.child_classes_models.append(self.disaggregation_model)
        self.disaggregation_model_file_prefix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '_disaggregation_model_' \
                                            + self.config['model_name'] + '_' + self.config['dataset_name'] + '_' + self.config['learning_style']
        self.child_classes_model_file_prefixes.append(self.disaggregation_model_file_prefix)

    # the training step for learning the easy way
    @tf.function
    def train_step(self, x, y):

        # one-hot encode the latent probabilities of the true target
        z_true_list = self.disaggregation_model(y, training=False)
        z_true_list = [tf.one_hot(tf.argmax(z, axis=1), depth=units)
                       for z, units in zip(z_true_list, self.num_disaggregator_units)]

        with tf.GradientTape(persistent=True) as tape:

            y_pred = self.prediction_model(x, training=True)
            loss_y = self.cat_loss(y_true=y, y_pred=y_pred)

            # define the loss of the disaggregation
            z_pred_list = self.disaggregation_model(y_pred, training=True)
            loss_z = tf.reduce_mean([self.disaggregation_loss(y_true=z_true, y_pred=z_pred) for z_true, z_pred in zip(z_true_list, z_pred_list)])

            if self.config['lew_mode'] == 'lew':
                loss_prediction_model = loss_y + loss_z
                loss_disaggregation_model = loss_z
            elif self.config['lew_mode'] == 'random':
                loss_prediction_model = loss_y + loss_z
            elif self.config['lew_mode'] == 'min':
                loss_prediction_model = loss_y
                loss_disaggregation_model = loss_z

        # update the prediction model
        prediction_model_weights = self.prediction_model.trainable_variables
        prediction_gradients = tape.gradient(loss_prediction_model, prediction_model_weights)
        self.prediction_optimizer.apply_gradients(zip(prediction_gradients, prediction_model_weights))

        # update the disaggregation model
        if self.config['lew_mode'] == 'lew' or self.config['lew_mode'] == 'min':
            disaggregation_model_weights = self.disaggregation_model.trainable_variables
            disaggregation_gradients = tape.gradient(loss_disaggregation_model, disaggregation_model_weights)
            self.disaggregation_optimizer.apply_gradients(zip(disaggregation_gradients, disaggregation_model_weights))

        # update the metrics
        self.train_loss(loss_y)
        self.train_accuracy(y, y_pred)
        self.disaggregation_loss_metric(loss_z)

