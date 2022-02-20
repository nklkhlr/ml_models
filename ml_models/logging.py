import numpy as np
from typing import List


class Logger:
    def __init__(self, n_epochs: int, batch_size: int):
        self._batch_size = batch_size
        self._n_epochs = n_epochs
        self.epoch_loss = []
        self.train_metric = []
        self.test_metric = np.zeros(n_epochs)
        self.test_loss = np.zeros(n_epochs)
        self.current_epoch = 0

    def reset(self):
        self.epoch_loss = []
        self.train_metric = []
        self.test_metric = np.zeros(self._n_epochs)
        self.test_loss = np.zeros(self._n_epochs)
        self.current_epoch = 0

    def update(self, loss: List[float], train_metric: List[float],
               test_metric: float, test_loss: float):
        self.epoch_loss.append((np.nanmean(loss), np.nanstd(loss)))
        self.train_metric.append((np.nanmean(test_metric), np.nanstd(train_metric)))
        self.test_metric[self.current_epoch] = test_metric
        self.test_loss[self.current_epoch] = test_loss
        self.current_epoch += 1
