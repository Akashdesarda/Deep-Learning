from keras.callbacks import BaseLogger, Callback, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import os
import json


class TrainingMonitor(BaseLogger):
    def __init(self, filePath, jsonPath=None, startAt=0):
        """It used to Monitor training (training loss/acc and val loss/acc after every epoch)

        Args:
            filePath (str): directory to save plots dynamically
            jsonPath (str, optional): directory to serialize training history into json.
            It can also overwrite previous json. Defaults to None.
            startAt (int, optional): Epoch no to resume training. Defaults to 0.
        """
        self.filePath = filePath
        self.jsonPath = jsonPath
        self.startAt = startAt

    def on_train_begin(self, logs={}):
        # init training history
        self.H = {}

        # if json path exit then load it
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.load(open(self.jsonPath).read())

                # check if starting epoch is suplied
                if self.startAt > 0:

                    # loop over the history dictionary to check last epoch and trim past entries
                    for i in self.H.keys():
                        self.H[i] = self.H[i][:self.startAt]

    def on_epoch_end(self, epoch, logs={}):
        # loop over the History logs to update acc, loss
        for (key, value) in logs.items():
            temp_list = self.H.get(key, [])
            temp_list.append(value)
            self.H = temp_list

        # if json path is provided then update data into it
        if self.jsonPath is not None:
            f = open(self.jsonPath, 'w')
            f.write(json.dump(self.H))
            f.close()

        # ensure at 2 epoch have passed & then plotting history
        if len(self.H['loss']) > 1:
            N = np.arange(0, len(self.H["loss"]))
            plt.figure()
            plt.plot(N, self.H["loss"], label="train_loss")
            plt.plot(N, self.H["val_loss"], label="val_loss")
            plt.plot(N, self.H["acc"], label="train_acc")
            plt.plot(N, self.H["val_acc"], label="val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(
                len(self.H["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()

            # save the figure
            plt.savefig(self.figPath)
            plt.close()

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
