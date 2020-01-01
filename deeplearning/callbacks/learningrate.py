import numpy as np

def step_decay(epoch):
    
    # Initialize base learning rate drop factor and epochs to drop every
    initAlpha = 0.01
    factor = 0.25
    dropEvery = 5 
    
    # compute learning rate for the current epoch 
    rate = initAlpha * (factor ** np.floor(1 + epoch)/dropEvery)
    return float(rate)