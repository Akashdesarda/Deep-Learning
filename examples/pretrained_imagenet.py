import argparse

import cv2
import numpy as np
from keras.applications import imagenet_utils
from keras.applications import inception_v3
from keras.applications import resnet50
from keras.applications import vgg16, vgg19
from keras.applications import xception
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img

# Constructing argument parse
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help='path to input image')
ap.add_argument('-m', '--model', type=str, default='resnet50',
                help='name of pre-trained network to use')
args = vars(ap.parse_args())

# Defining a dict that maps model names
MODELS = {
    'vgg16': vgg16,
    'vgg19': vgg19,
    'inception_v3': inception_v3,
    'xception': xception,
    'resnet50': resnet50
}
# ensuring a valid model name is passed
if args['model'] not in MODELS.keys():
    raise AssertionError("Model name provided should be from "
                         "keys in MODEL dict")

# Initialising the image input shape to (224,224)
inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input

# If model are Inception or Xception network then input shape will (299, 299)
if args['model'] in ('inception_v3', 'xception'):
    inputShape = (299, 299)
    preprocess = preprocess_input  # a different preprocess method for them taken from inception.preprocess_input

# loading the network
print("[WARNING]... If this script is run for first time then the weights will be downloaded first"
      "So be patient")
print("[INFO] loading {}...".format(args['model']))
Network = MODELS[args['model']]
model = Network(weigts='imagenet')

print("[INFO] loding and preprocessing image")
image = load_img(args['image'], target_size=inputShape)
image = img_to_array(image)

# Making the dimension as (1, inputShape[0), inputShape[1], 3)
image = np.expand_dims(image, axis=0)

# transforming image according to the needs
image = preprocess(image)
print("[INFO] classifying given image using {} model...".format(args['model']))
pred = model.predict(image)
P = imagenet_utils.decode_predictions(pred)

for (i, (imagenetID, label)) in enumerate(P[0]):
    print("{}. {}".format(i + 1, label))

ori = cv2.imread(args['image'])
(imagenetID, label, prob) = P[0][0]
cv2.putText(ori, "Label: {}".format(label), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.imshow("Classification", ori)
cv2.waitKey(0)
