from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential


class MinivVGGNet:

    @staticmethod
    def build(width, height, depth, classes):
        """
        Input image details and no of classes also assuming 'channel_last'
        eg build(227,227,3)
        :param width: [int] width of image
        :param height: [int] height of image
        :param depth: [int]
        :param classes: [int] Total no of classes
        :return: MiniVGGNet model
        """
        model = Sequential()
        inputShape = (width, height, depth)

        # 1st layer conv => relu =>BN=> conv =>relu=>BN=>pool=>DO
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())
        model.add(Dropout(0.25))

        # 2nd layer conv => relu =>BN=> conv =>relu=>BN=>pool=>DO
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())
        model.add(Dropout(0.25))

        # FC layer FC=>relu=>BN=>DO
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes, activation='softmax'))
        return model
