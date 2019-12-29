import os
import cv2
import numpy as np
from typing import List


class SimpleDatasetLoader:
    def __init__(self, preprocessors: List=None):
        self.preprocessors = preprocessors

        #if preprocessors is None then initialize as empty list
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths: str, verbose: int=-1):
        """Load images and extract label name assuming path will be as:
           path/to/image/class/{image}.jpg

        Arguments:
            imagePaths [str] -- Path to the images

        Keyword Arguments:
            verbose {int} -- if < 0 then will show detail info (default: {-1})
            
        Returns:
            data {np.array} -- image data in numpy array 
            labels {str} -- class name of respective image
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
                print(f"[INFO] processed image {i + 1}/{len(imagePath)}")

        return (np.array(data), np.array(labels))
