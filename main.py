from image_classification.DataInterface import DataInterface
from image_classification.PredictionModel import PredictionModel
from image_classification.DefaultModelOptimizer import DefaultModelOptimizer
import argparse
import tensorflow as tf



parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="The name of the tfds dataset, e.g. {mnist, cifar10}")
parser.add_argument("model", help="The name of the model, e.g. {DenseNet121, NASNetMobile}")
parser.add_argument('--epochs', help='Number of epochs', type=int)
parser.add_argument('--batch_size', help='Batch size', type=int)
parser.add_argument('--eta', help='Learning rate', type=float)
parser.add_argument('--div_reg', help='Diversity penalty', type=float)
parser.add_argument('--ensemble_type', help='simple or scalable')
parser.add_argument('--image_size', help='Input image size', nargs='+', type=int)
parser.add_argument("--checkpoints_load_prefix", help="The path of the dataset from which to init the checkpoints")
parser.add_argument("--checkpoints_save_prefix", help="The path of the dataset where to save the checkpoints")

args = parser.parse_args()

config = {'buffer_size': 10000,
          'image_size': (224, 224, 3),
          'num_epochs': 1001,
          'eta': 0.001,
          'batch_size': 10000,
          'test_frequency': 1,
          'checkpoint_frequency': 10}

# the default image sizes that the models expect


if args.model == 'MobileNetV2':
    config['image_size'] = (224, 224, 3)
elif args.model == 'MobileNetV3Small' or args.model == 'MobileNetV3Large':
    config['image_size'] = (224, 224, 3)
elif args.model == 'NASNetMobile':
    config['image_size'] = (224, 224, 3)
elif args.model == 'DenseNet201':
    config['image_size'] = (224, 224, 3)
elif args.model == 'Xception':
    config['image_size'] = (299, 299, 3)
elif args.model == 'ResNet50V2':
    config['image_size'] = (224, 224, 3)
elif args.model == 'InceptionV3':
    config['image_size'] = (299, 299, 3)
elif args.model == 'EfficientNetB0':
    config['image_size'] = (224, 224, 3)
elif args.model == 'EfficientNetB3':
    config['image_size'] = (300, 300, 3)
else:
    config['image_size'] = (32, 32, 3)

if args.image_size:
    config['image_size'] = args.image_size


config['batch_size'] = args.batch_size

# load the dataset
data_interface = DataInterface(config=config)
data_interface.load(args.dataset)

# create the prediction model
models = []
pm = PredictionModel(config=config)
m = pm.create_prediction_model(model_name=args.model, num_classes=data_interface.num_classes)
m.summary()

de = DefaultModelOptimizer(prediction_model=m, config=config, data_interface=data_interface)
de.run()

