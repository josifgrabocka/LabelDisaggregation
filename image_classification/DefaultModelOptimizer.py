import tensorflow as tf
import time
import tensorflow_addons as tfa

class DefaultModelOptimizer:

    def __init__(self, prediction_model, data_interface, config):
        self.prediction_model = prediction_model
        self.data_interface = data_interface
        self.train_ds = self.data_interface.train_ds
        self.test_ds = self.data_interface.test_ds
        self.num_classes = self.data_interface.num_classes
        self.config = config
        self.l2_penalty=config["l2_penalty"]

        first_decay_steps = 0
        for _ in self.train_ds:
            first_decay_steps += 1

        # create the initializers
        # the cosine decay learning rate scheduler with restarts and the decoupled L2 adam with gradient clipping
        step = tf.Variable(0, trainable=False)
        lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=config['eta'],
                                                                     first_decay_steps=first_decay_steps)
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
        self.checkpoint_frequency = config['checkpoint_frequency']
        # a prefix string for the checkpoint file name, e.g. dataset name, model name
        # if needed to be set outside outside the class after the constructor
        self.checkpoint_prefix = ''

    # the training step of the prediction model
    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            y_pred = self.prediction_model(x, training=True)
            loss_pred = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=y, y_pred=y_pred, from_logits=True))

        weights = self.prediction_model.trainable_variables
        prediction_gradients = tape.gradient(loss_pred, weights)
        self.prediction_optimizer.apply_gradients(zip(prediction_gradients, weights))

        self.train_loss(loss_pred)
        self.train_accuracy(y, y_pred)

    # the testing step
    @tf.function
    def test_step(self, x, y):
        y_pred = self.prediction_model(x, training=False)
        t_loss = tf.keras.losses.categorical_crossentropy(y_true=y, y_pred=y_pred, from_logits=True)
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

                template = 'Epoch {}, Train: {:4.4f}, {:4.4f}, Test: {:4.4f}, {:4.4f}, Time: {:4.4f}'
                print(template.format(epoch + 1,
                                      self.train_loss.result(),
                                      self.train_accuracy.result(),
                                      self.test_loss.result(),
                                      self.test_accuracy.result(),
                                      time.time()-start_time))

                self.train_loss.reset_states()
                self.train_accuracy.reset_states()
                self.test_loss.reset_states()
                self.test_accuracy.reset_states()

            #if epoch % self.checkpoint_frequency == 0:
            #    # save the checkpoints of the prediction and attack models
            #    self.prediction_model.save_weights('./checkpoints/prediction_' + self.checkpoint_prefix + '.h5')
