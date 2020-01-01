import numpy as np

class Custom_lr:

    def __init__(self, initAlpha: float=0.01, factor: float=0.25, dropEvery: int=5):
        """Custom Learning rate
        Arguments:
            initAlpha {float} -- To initialize learning rate
            factor {float} -- Factor by which drop it
            dropEvery {int} -- Drop every epoch no
        """
        self.initAlpha = initAlpha
        self.factor = factor
        self.dropEvery = dropEvery
        
    def step_decay(self,epoch):    
    # compute learning rate for the current epoch 
        rate = self.initAlpha * (self.factor ** np.floor(1 + epoch)/self.dropEvery)
        print(f"Using Step Decay Learning rate with specs as initAlpha: {self.initAlpha}, factor: {self.factor}, dropEvery:{self.dropEvery}")
        return float(rate)