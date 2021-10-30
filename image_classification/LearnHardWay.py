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
        disaggregation_outputs = []
        for frac in config['disaggregation_layers_fracs']:
            h = tf.keras.layers.Dense(units=int(frac*self.data_interface.num_classes), activation='sigmoid')(h)
            disaggregation_outputs.append(h)
        self.disaggregation_model = tf.keras.Model(inputs=disaggregator_input, outputs=disaggregation_outputs)
        self.disaggregation_model.summary()

        #define the additional loss terms metrics
        self.disaggregation_loss = tf.keras.metrics.Mean(name='disaggregation_loss')
        self.logs_metrics.append(self.disaggregation_loss)
        # add the additional loss term definitions
        self.bin_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        # the cosine decay learning rate scheduler with restarts and the decoupled L2 adam with gradient clipping
        step = tf.Variable(0, trainable=False)
        lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=config['eta'],
                                                                     t_mul=1,
                                                                     first_decay_steps=self.first_decay_steps)
        wd = self.l2_penalty * lr_sched(step)
        self.disaggregation_optimizer = tfa.optimizers.AdamW(learning_rate=lr_sched, weight_decay=wd)

        # populate the list of models added by child classes of the DefaultOptimizer, such that the saving of these models
        # can happen inside the run method of the parent class
        self.child_classes_models.append(self.disaggregation_model)
        self.disaggregation_model_file_prefix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '_disaggregation_model_' \
                                            + self.config['model_name'] + '_' + self.config['dataset_name'] + '_' + self.config['learning_style']
        self.child_classes_model_file_prefixes.append(self.disaggregation_model_file_prefix)

    # the training step for learning the hard way
    @tf.function
    def train_step(self, x, y):

        # binarize the targets
        z_true_list = self.disaggregation_model(y, training=False)
        z_true_list = [tf.round(z) for z in z_true_list]

        with tf.GradientTape(persistent=True) as tape:

            y_pred = self.prediction_model(x, training=True)
            loss_y = self.cat_loss(y_true=y, y_pred=y_pred)

            # define the loss of the disaggregation
            z_pred_list = self.disaggregation_model(y_pred, training=True)
            loss_z = tf.reduce_mean([self.bin_loss(y_true=z_true, y_pred=z_pred) for z_true, z_pred in zip(z_true_list, z_pred_list)])

            if self.config['lhw_mode'] == 'lhw':
                loss_prediction_model = loss_y + loss_z
                loss_disaggregation_model = -tf.sigmoid(loss_z)
            elif self.config['lhw_mode'] == 'random':
                loss_prediction_model = loss_y + loss_z
            elif self.config['lhw_mode'] == 'max':
                loss_prediction_model = loss_y
                loss_disaggregation_model = -tf.sigmoid(loss_z)

        # update the prediction model
        prediction_model_weights = self.prediction_model.trainable_variables
        prediction_gradients = tape.gradient(loss_prediction_model, prediction_model_weights)
        self.prediction_optimizer.apply_gradients(zip(prediction_gradients, prediction_model_weights))

        # update the disaggregation model
        if self.config['lhw_mode'] == 'lhw' or self.config['lhw_mode'] == 'max':
            disaggregation_model_weights = self.disaggregation_model.trainable_variables
            disaggregation_gradients = tape.gradient(loss_disaggregation_model, disaggregation_model_weights)
            self.disaggregation_optimizer.apply_gradients(zip(disaggregation_gradients, disaggregation_model_weights))
            self.disaggregation_loss(loss_z)

        # update the metrics
        self.train_loss(loss_y)
        self.train_accuracy(y, y_pred)

