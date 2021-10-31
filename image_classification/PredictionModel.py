import tensorflow as tf

class PredictionModel:

    def __init__(self, config):
        self.config = config
        self.embedding = None
        self.prediction_model = None
        self.default_input_size = config['image_size']

    # create the prediction model
    def create_embedding_model(self, model_name):

        weights='imagenet'
        pretrained_embedding = None
        if model_name == 'MobileNetV2':
            pretrained_embedding = tf.keras.applications.mobilenet_v2.MobileNetV2(weights=weights,
                                                                                  include_top=False,
                                                                                  pooling='avg',
                                                                                  input_shape=self.default_input_size)
        elif model_name == 'MobileNetV3Small':
            pretrained_embedding = tf.keras.applications.MobileNetV3Small(include_top=False,
                                                                          weights=weights,
                                                                          pooling='avg',
                                                                          input_shape=self.default_input_size)
            # to do fix the dimension 1 axes in the output tensor
        elif model_name == 'MobileNetV3Large':
            pretrained_embedding = tf.keras.applications.MobileNetV3Large(weights=weights,
                                                                          pooling='avg',
                                                                          include_top=False,
                                                                          input_shape=self.default_input_size)
            # to do fix the dimension 1 axes in the output tensor


        elif model_name == 'NASNetMobile':
            pretrained_embedding = tf.keras.applications.nasnet.NASNetMobile(weights=weights,
                                                                             include_top=False,
                                                                             pooling='avg',
                                                                             input_shape=self.default_input_size)
        elif model_name == 'DenseNet121':
            pretrained_embedding = tf.keras.applications.densenet.DenseNet121(weights=weights,
                                                                              include_top=False,
                                                                              pooling='avg',
                                                                              input_shape=self.default_input_size)
        elif model_name == 'Xception':
            pretrained_embedding = tf.keras.applications.xception.Xception(include_top=False,
                                                                           weights=weights,
                                                                           pooling='avg',
                                                                           input_shape=self.default_input_size)
        elif model_name == 'ResNet50V2':
            pretrained_embedding = tf.keras.applications.resnet_v2.ResNet50V2(include_top=False,
                                                                              weights=weights,
                                                                              pooling='avg',
                                                                              input_shape=self.default_input_size)
        elif model_name == 'InceptionV3':
            pretrained_embedding = tf.keras.applications.inception_v3.InceptionV3(include_top=False,
                                                                                  weights=weights,
                                                                                  pooling='avg',
                                                                                  input_shape=self.default_input_size)
        elif model_name == 'EfficientNetB0':
            pretrained_embedding = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False,
                                                                                     weights=weights,
                                                                                     pooling='avg',
                                                                                     input_shape=self.default_input_size)
        elif model_name == 'EfficientNetB3':
            pretrained_embedding = tf.keras.applications.efficientnet.EfficientNetB3(include_top=False,
                                                                                     weights=weights,
                                                                                     pooling='avg',
                                                                                     input_shape=self.default_input_size)
        elif model_name == 'mini':
            num_filters = 16
            kernel_size = 3
            pool_size = 2

            pretrained_embedding = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size, activation='relu',
                                       input_shape=self.default_input_size, padding='same'),
                tf.keras.layers.AvgPool2D(pool_size),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size, activation='relu', padding='same'),
                tf.keras.layers.AvgPool2D(pool_size),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size, activation='relu', padding='same'),
                tf.keras.layers.AvgPool2D(pool_size),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size, activation='relu', padding='same'),
                tf.keras.layers.AvgPool2D(pool_size),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size, activation='relu', padding='same'),
                tf.keras.layers.Flatten()])

        return pretrained_embedding
        
    def create_prediction_model(self, model_name, num_classes):

        embedding = self.create_embedding_model(model_name)

        model_input = tf.keras.Input(self.default_input_size)
        # yield the representation model
        latent_representation = embedding(model_input)#
        # remove dimensions of 1 from the representation tensor, e.g. MobileNetV3 returns None,1,1,1024 and we convert it to None,1024
        #latent_representation = tf.squeeze(latent_representation, axis=[1, 2])
        y_pred = tf.keras.layers.Dense(num_classes, activation='softmax')(latent_representation)
        self.prediction_model = tf.keras.Model(inputs=model_input, outputs=y_pred)

        return self.prediction_model
