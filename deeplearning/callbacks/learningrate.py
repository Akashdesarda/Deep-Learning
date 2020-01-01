import numpy as np

def step_decay(epoch, initAlpha: float=0.01, factor: float=0.25, dropEvery: int=5):
    """Step decay Learning rate
    
    Arguments:
        initAlpha {float} -- To initialize learning rate
        factor {float} -- Factor by which drop it
        dropEvery {int} -- Drop every epoch no
    """
    # compute learning rate for the current epoch 
    rate = initAlpha * (factor ** np.floor(1 + epoch)/dropEvery)
    print(f"Using Step Decay Learning rate with specs as initAlpha: {initAlpha}, factor: {factor}, dropEvery:{dropEvery}")
    return float(rate)