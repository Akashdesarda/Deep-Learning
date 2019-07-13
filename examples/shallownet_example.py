import argparse
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from deeplearning.preprocessing import ImageToArrayPreprocessor
from deeplearning.preprocessing import SimplePreprocessor
from deeplearning.datasets import SimpleDatasetLoader
from deeplearning.nn.conv import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(seed=42)

# Constructing the argument parser and then parsing the argument in it
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
ap.add_argument('-c', '--classes', required=True, help='Total no of classes')
ap.add_argument('-b', '--bath_size', required=True, help='Batch size for network to train')
ap.add_argument('-e', '--epoch', required=True, help='No of epoch on which network will train')
args = vars(ap.parse_args())

# Will grab images inside the given dataset
print('[INFO] loading images inside given dataset')
imagePaths = list(paths.list_images(args['dataset']))

# Initialize image preprocessing
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

# load data from dataset & then convert into given pixel intensity
dl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = dl.load(imagePaths, verbose=1)
data = data.astype('float') / 255.0

(x_train, x_test, y_train, y_test) = train_test_split(data, labels,
                                                      test_size=0.25, random_state=42)
y_train = LabelBinarizer.fit_transform(y_train)
y_test = LabelBinarizer.fit_transform(y_test)

print('[INFO] compiling model...')
opt = SGD(lr=0.00105)
model = ShallowNet.build(width=32, height=32, depth=3, classes=int(args['classes']))
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# training model
print('[INFO] training network')
History = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                    batch_size=int(args['batch_size']), epochs=int(args['epoch']))

# Evaluating model
print('[INFO] Evaluating network...')
pred = model.predict(x_test, batch_size=int(args['batch_size']))
print(classification_report(y_test.argmax(axis=1), pred.argmax(axis=1), target_names=labels))

# plot the training and accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, int(args['epoch'])), History.history['loss'], label='train_loss')
plt.plot(np.arange(0, int(args['epoch'])), History.history['val_loss'], label='val_loss')
plt.title('Training loss and accuracy')
plt.xlabel('Epoch no')
plt.ylabel('Loss')
plt.legend()
plt.show()