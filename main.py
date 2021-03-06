from image_classification.DataInterface import DataInterface
from image_classification.PredictionModel import PredictionModel
from image_classification.DefaultOptimizer import DefaultOptimizer
from image_classification.LearnHardWay import LearnHardWay
import argparse
import tensorflow as tf



parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="The name of the tfds dataset, e.g. {mnist, cifar10}")
parser.add_argument("model", help="The name of the model, e.g. {DenseNet121, NASNetMobile}")
parser.add_argument('--learning_style', help='Style: {easy, normal}')
parser.add_argument('--epochs', help='Number of epochs', type=int)
parser.add_argument('--batch_size', help='Batch size', type=int)
parser.add_argument('--eta', help='Learning rate', type=float)
parser.add_argument('--gamma', help='Disaggregation loss penalty', type=float)
parser.add_argument('--lambda', help='L2 penalty', type=float)
parser.add_argument('--lhw_mode', help='lhw, random, max')
parser.add_argument('--image_size', help='Input image size', nargs='+', type=int)
parser.add_argument("--checkpoints_load_prefix", help="The path of the dataset from which to init the checkpoints")
parser.add_argument("--checkpoints_save_prefix", help="The path of the dataset where to save the checkpoints")

args = parser.parse_args()

config = {'buffer_size': 5000,
          'image_size': (224, 224, 3),
          'num_epochs': 200,
          'eta': 0.00003,
          'gamma': 10.0,
          'batch_size': 100,
          'test_frequency': 1,
          'l2_penalty': 0.0001,
          'checkpoint_frequency': 20}

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

# set the configurations
if args.image_size:
    config['image_size'] = args.image_size
if args.learning_style:
    config['learning_style'] = args.learning_style
if args.batch_size:
    config['batch_size'] = args.batch_size
# write the model and dataset name to the config
config['model_name'] = args.model
config['dataset_name'] = args.dataset


# load the dataset
data_interface = DataInterface(config=config)
data_interface.load(args.dataset)
# set gamma proportional to the number of classes
print('Gamma set to', config['gamma'])



# create the prediction model
models = []
pm = PredictionModel(config=config)
m = pm.create_prediction_model(model_name=args.model, num_classes=data_interface.num_classes)
m.summary()

# run the optimizer
if config['learning_style'] == 'normal':
    de = DefaultOptimizer(prediction_model=m, config=config, data_interface=data_interface)
    de.run()
elif config['learning_style'] == 'hard':
    config['disaggregation_layers_fracs'] = [1.0, 1.0, 1.0]
    if args.lhw_mode:
        config['lhw_mode'] = args.lhw_mode
    lew = LearnHardWay(prediction_model=m, config=config, data_interface=data_interface)
    lew.run()
