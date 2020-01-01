import argparse
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import tensorflow as tf
from keras.callbacks import LearningRateScheduler
from keras.datasets import cifar10
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from timeit import default_timer as timer

from deeplearning.nn.conv import MinivVGGNet
from deeplearning.callbacks.learningrate import custom_lr
lr = custom_lr(initAlpha =0.01, factor=0.25, dropEvery =5)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Constructing the argument parser and then parsing the argument in it
parser = argparse.ArgumentParser()
# parser.add_argument('-d', '--dataset', required=True, help='path to input dataset')
parser.add_argument('-c', '--classes', required=True, help='Total no of classes')
parser.add_argument('-b', '--batch_size', required=True, help='Batch size for network to train')
parser.add_argument('-e', '--epoch', required=True, help='No of epoch on which network will train')
parser.add_argument('-l', '--limit_gpu_usage', default=True, help='Enable limiting gpu memory graph')
# device_name = sys.argv[5]
args = vars(parser.parse_args())

start = timer()
# Will grab images inside the given dataset
print('[INFO] loading images inside given dataset')
# imagePaths = list(paths.list_images(args['dataset']))
#
# # Initialize image preprocessing
# sp = SimplePreprocessor(32, 32)
# iap = ImageToArrayPreprocessor()
#
# # load data from dataset & then convert into given pixel intensity
# dl = SimpleDatasetLoader(preprocessors=[sp, iap])
# (data, labels) = dl.load(imagePaths, verbose=1)
# data = data.astype('float') / 255.0

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = LabelBinarizer().fit_transform(y_train)
y_test = LabelBinarizer().fit_transform(y_test)

callback = [LearningRateScheduler(lr.step_decay()]
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print('[INFO] compiling model...')
opt = SGD(lr=0.001, momentum=0.9, decay=0.001 / int(args['epoch']), nesterov=True)
model = MinivVGGNet.build(width=32, height=32,
                          depth=3, classes=int(args['classes']))
model.compile(loss='categorical_crossentropy', optimizer=opt,
              metrics=['accuracy'])

# Checkpoint to save best model
# save_model = ModelCheckpoint(filepath=, monitor='val_loss', save_best_only=True, mode='auto')
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
History = model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=callback,
                    batch_size=int(args['batch_size']), epochs=int(args['epoch']))

# Evaluating model
print('[INFO] Evaluating network...')
pred = model.predict(x_test, batch_size=int(args['batch_size']))
print(classification_report(y_test.argmax(axis=1), pred.argmax(axis=1), target_names=labels))

# plot the training and accuracy
# plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, int(args['epoch'])), History.history['loss'], label='train_loss')
plt.plot(np.arange(0, int(args['epoch'])), History.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, int(args['epoch'])), History.history['accuracy'], label='train_acc')
plt.plot(np.arange(0, int(args['epoch'])), History.history['val_accuracy'], label='val_acc')
plt.title('Training loss and accuracy')
plt.xlabel('Epoch no')
plt.ylabel('Loss/Accuracy')
plt.legend()
end = timer()
print(f"Total Time taken is {start - end}/60:.5 min(s)")

plt.show()