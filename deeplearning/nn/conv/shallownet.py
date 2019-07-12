from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Activation, Flatten


class ShallowNet:

    @staticmethod
    def build(width, height, depth, classes):
        """
        Input image details and no of classes also assuming 'channel_last'
        eg build(227,227,3)
        :param width: [int] width of image
        :param height: [int] height of image
        :param depth: [int]
        :param classes: [int] Total no of classes
        :return: Convulation ShallowNet model
        """
        model = Sequential()
        inputShape = (width, height, depth)

        # defining 1st layer : conv => relu
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=inputShape))
        model.add(Activation('relu'))

        # softmax classifier
        model.add(Flatten())
        model.add(Dense(classes, activation='softmax'))
        return model
