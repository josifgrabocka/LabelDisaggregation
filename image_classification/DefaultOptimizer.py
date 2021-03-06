import tensorflow as tf
import time
import tensorflow_addons as tfa
import datetime

class DefaultOptimizer:

    def __init__(self, prediction_model, data_interface, config):
        self.prediction_model = prediction_model
        self.data_interface = data_interface
        self.train_ds = self.data_interface.train_ds
        self.test_ds = self.data_interface.test_ds
        self.num_classes = self.data_interface.num_classes
        self.config = config
        self.l2_penalty=config["l2_penalty"]

        self.num_restart_epochs = 20
        self.first_decay_steps = 0
        for _ in self.train_ds:
            self.first_decay_steps += 1
        # make the first steps for X=10 epochs
        self.first_decay_steps *= self.num_restart_epochs

        # create the initializers
        # the cosine decay learning rate scheduler with restarts and the decoupled L2 adam with gradient clipping
        step = tf.Variable(0, trainable=False)
        lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=config['eta'],
                                                                     t_mul=1.0,
                                                                     first_decay_steps=self.first_decay_steps)
        wd = self.l2_penalty * lr_sched(step)

        self.prediction_optimizer = tfa.optimizers.AdamW(learning_rate=lr_sched, weight_decay=wd)

        # the metrics for storing the training performance
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

        # metrics for storing testing performance
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

        self.num_epochs = config['num_epochs']
        self.test_frequency = config['test_frequency']
        self.save_model_frequency = config['checkpoint_frequency']
        # a prefix string for the checkpoint file name, e.g. dataset name, model name
        # if needed to be set outside outside the class after the constructor
        self.prediction_model_file_prefix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '_prediction_model_' \
                                            + self.config['model_name'] + '_' + self.config['dataset_name'] + '_' + self.config['learning_style']
        self.save_models = False

        self.logs_metrics = [self.train_loss, self.train_accuracy, self.test_loss, self.test_accuracy]

        # define loss functions
        self.cat_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

        # other models that might be used by child classes
        self.child_classes_models = []
        self.child_classes_model_file_prefixes = []

    # the training step of the prediction model
    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            y_pred = self.prediction_model(x, training=True)
            loss_pred = tf.reduce_mean(self.cat_loss(y_true=y, y_pred=y_pred))

        weights = self.prediction_model.trainable_variables
        prediction_gradients = tape.gradient(loss_pred, weights)
        self.prediction_optimizer.apply_gradients(zip(prediction_gradients, weights))

        self.train_loss(loss_pred)
        self.train_accuracy(y, y_pred)

    # the testing step
    @tf.function
    def test_step(self, x, y):
        y_pred = self.prediction_model(x, training=False)
        t_loss = self.cat_loss(y_true=y, y_pred=y_pred)
        self.test_loss(t_loss)
        self.test_accuracy(y, y_pred)

    # the training algorithm
    def run(self):

        start_time = time.time()

        for epoch in range(self.num_epochs):
            for x, y in self.train_ds:
                self.train_step(x, y)

            # test the model's performance at a specified frequency
            if epoch % self.test_frequency == 0:
                # measure the test accuracy
                for x, y in self.test_ds:
                    self.test_step(x, y)

                # print the metrics
                print('{},'.format(epoch), end='')
                for metric in self.logs_metrics:
                    print('{:4.4f},'.format(metric.result().numpy()), end='')
                print('{:4.2f}'.format(time.time() - start_time))

                # reset the metrics
                for metric in self.logs_metrics:
                    metric.reset_states()

            # save the checkpoints
            if self.save_models:
                if (epoch+1) % self.save_model_frequency == 0:
                    # save the checkpoints of the prediction and attack models
                    self.prediction_model.save_weights('./logs/models/' + self.prediction_model_file_prefix + '_' + str(epoch) + '.h5')
                    # save other models in the list that might be populated by child classes
                    for m, prefix in zip(self.child_classes_models, self.child_classes_model_file_prefixes):
                        m.save_weights('./logs/models/' + prefix + '_' + str(epoch) + '.h5')

                    # save models for the snapshot ensemble