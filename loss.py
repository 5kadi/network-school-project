import numpy as np
from config import EPSILON

CLASSES = [i for i in range(10)]

class CrossEntropyLoss():
    def __call__(self, pred: np.ndarray, ref: np.ndarray) -> np.float64:
        pred_e = pred + EPSILON
        self.loss = -1 * np.sum(ref * np.log(pred_e)) #log = ln

        #dE/dsi = -1 * yi / si
        self.grad = -1 * ref / pred_e

        return self.loss 



#print(np.sum(pred, 0))
