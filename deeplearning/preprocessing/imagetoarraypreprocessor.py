# Importing necessary packages
from keras.preprocessing.image import img_to_array


class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        self.dataFormat = dataFormat

    def preprocess(self, image):
        """Takes an image to convert it into numpy array and apply the Keras utility function that correctly rearranges

        Args:
            image: Can be path of image

        Returns:
            np.array: np.array of image
        """
        return img_to_array(image, data_format=self.dataFormat)
