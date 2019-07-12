import os
from ntpath import split

import cv2
import numpy as np


class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors

        #if preprocessors is None then initialize as empty list
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        """Load images and extract label name assuming path will be as:
           path/to/image/class/{image}.jpg

        Arguments:
            imagePaths [str] -- Path to the images

        Keyword Arguments:
            verbose {int} -- if < o then will show detail info (default: {-1})
        """
        data = []
        labels = []

        for (i, imagePath) in enumerate(imagePaths):
            #Load image & extract class name make sure path must be
            # path/to/image/class/{image}.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)

            data.append(image)
            labels.append(label)

            #show verbose after every image
            if verbose > 0 and i > 0:
                print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))

        return (np.array(data), np.array(labels))

