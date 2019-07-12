import cv2

class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        """Takes Image as input

        Arguments:
            width {int} -- width of image
            height {int} -- height of image

        """
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        """Resize the image to fixed size, ignoring the original aspect ratio

        Arguments:
            image {np.array} -- Image which is converted into np.array
        """
        return cv2.resize(image, (self.width, self.height),interpolation = self.inter)

