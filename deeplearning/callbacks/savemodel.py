from keras.callbacks import ModelCheckpoint


class SaveModel:
     def __init__(self, filePath, monitorMetric='val_loss', saveMultiple=False):
         """Callback to save model at suplied directory at respective checkpoint

         Args:
             filePath (str): directory path to save model.
             monitorMetric (str, optional): Metric to used for saving model. Defaults to 'val_loss'. It can be also 'loss','accuracy','val_accuracy'
             saveMultiple (bool, optional): To save only single model or every better model. Defaults to 'False' to save only single best model and 'True' to save every better model.
         """
        self.filePath = filePath
        self.monitorMetric = monitorMetric
        self.saveMultiple = saveMultiple
        
        # callback to save model
        if self.saveMultiple is False:
            checkpoint = ModelCheckpoint(filepath=self.filePath, monitor= self.monitorMetric, mode="auto",save_best_only=True, verbose=1)

        elif self.saveMultiple is True:
            fname = os.path.sep.join([self.filePath,"weights-{epoch:03d}-{val_loss:.4f}.hdf5"])
            checkpoint = ModelCheckpoint(filepath=fname, monitor= self.monitorMetric, mode="auto",save_best_only=True, verbose=1)
        return checkpoint