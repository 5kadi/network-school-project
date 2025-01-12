import numpy as np

class BaseLoss:
    def __init__(self, *args) -> None: ...
    def __call__(self, pred: np.ndarray, ref: np.ndarray) -> np.float32: ...

class CrossEntropyLoss(BaseLoss):
    def __init__(self, epsilon: float) -> None:
        self.epsilon = epsilon

    def __call__(self, pred: np.ndarray, ref: np.ndarray) -> np.float64:
        pred_e = pred + self.epsilon #чтобы не было деления на 0
        self.loss = -1 * np.sum(ref * np.log(pred_e)) #log = ln
        #dE/dsi = -1 * yi / si
        self.grad = -1 * ref / pred_e
        return self.loss 
