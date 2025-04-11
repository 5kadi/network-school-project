import numpy as np
from abc import ABC, abstractmethod
from config import EPSILON

class AbstractLoss(ABC):
    @abstractmethod
    def __call__(self, pred: np.ndarray, ref: np.ndarray) -> np.float64: ...
    @abstractmethod
    def calculate_grad(self, *args, **kwargs) -> None: ...

class CrossEntropyLoss(AbstractLoss):
    def __call__(self, pred: np.ndarray, ref: np.ndarray) -> np.float64:
        pred_e = pred + ((np.abs(pred) < EPSILON) * EPSILON)
        loss = -1 * np.sum(ref * np.log(pred_e), -1, keepdims=True)
        self.calculate_grad(pred_e, ref)
        return loss 
    
    def calculate_grad(self, pred: np.ndarray, ref: np.ndarray) -> None:
        self.grad = -1 * ref / pred

"""
class MSELoss(AbstractLoss):
    def __call__(self, pred: np.ndarray, ref: np.ndarray) -> np.float64:
        loss = np.sum((pred - ref)**2, -1) / pred.shape[-1]
        self.calculate_grad(pred, ref)
        return loss
    
    def calculate_grad(self, pred: np.ndarray, ref: np.ndarray) -> None:
        self.grad = 2*(pred - ref) / pred.shape[-1]
"""