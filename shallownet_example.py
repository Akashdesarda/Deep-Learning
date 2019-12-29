import warnings
warnings.filterwarnings("ignore")

import argparse
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from deeplearning.preprocessing import ImageToArrayPreprocessor
from deeplearning.preprocessing import SimplePreprocessor
from deeplearning.datasets import SimpleDatasetLoader
from deeplearning.nn.conv import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

np.random.seed(seed=42)

# Constructing the argument parser and then parsing the argument in it
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, required=True, help='path to input dataset')
parser.add_argument('-c', '--classes', type=int, required=True, help='Total no of classes')
parser.add_argument('-b', '--batch_size', type=int, required=True, help='Batch size for network to train')
parser.add_argument('-e', '--epoch', type=int, required=True, help='No of epoch on which network will train')
parser.add_argument('-l', '--limit_gpu_usage', default=True, help='Enable limiting gpu memory graph')
args = vars(parser.parse_args())

start = timer()
# Will grab images inside the given dataset
print('[INFO] loading images inside given dataset')
imagePaths = list(paths.list_images(args['dataset']))

# Initialize image preprocessing
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

# load data from dataset & then convert into given pixel intensity
dl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = dl.load(imagePaths)
data = data.astype('float') / 255.0

(x_train, x_test, y_train, y_test) = train_test_split(data, labels,
                                                      test_size=0.25, random_state=42)
y_train = LabelBinarizer().fit_transform(y_train)
y_test = LabelBinarizer().fit_transform(y_test)

print('[INFO] compiling model...')
opt = SGD(lr=0.00105)
model = ShallowNet.build(width=32, height=32, depth=3, classes=int(args['classes']))
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Limiting GPU memory growth
if args['limit_gpu_usage'] is True:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

# training model
print('[INFO] training network')
History = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                    batch_size=int(args['batch_size']), epochs=int(args['epoch']))

# Evaluating model
print('[INFO] Evaluating network...')
pred = model.predict(x_test, batch_size=int(args['batch_size']))
print(classification_report(y_test.argmax(axis=1), pred.argmax(axis=1), target_names=['cat', 'dog', 'pandas']))

end = timer()
print(f"[INFO] Total time taken is {(end - start)/60:.4} min(s)")
# plot the training and accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, int(args['epoch'])), History.history['loss'], label='train_loss')
plt.plot(np.arange(0, int(args['epoch'])), History.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, int(args['epoch'])), History.history['accuracy'], label='train_acc')
plt.plot(np.arange(0, int(args['epoch'])), History.history['val_accuracy'], label='val_acc')
plt.title('Training loss and accuracy')
plt.xlabel('Epoch no')
plt.ylabel('Loss')
plt.legend()
plt.show()
