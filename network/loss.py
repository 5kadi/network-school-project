import numpy as np
from abc import ABC, abstractmethod
from config import EPSILON

class AbstractLoss(ABC):
    @abstractmethod
    def __call__(self, pred: np.ndarray, ref: np.ndarray) -> np.float32: ...
    @abstractmethod
    def calculate_grad(self, *args, **kwargs) -> None: ...

class CrossEntropyLoss(AbstractLoss):
    def __call__(self, pred: np.ndarray, ref: np.ndarray) -> np.float32:
        pred_e = pred + ((np.abs(pred) < EPSILON) * EPSILON)
        loss = -1 * np.sum(ref * np.log(pred_e))
        self.calculate_grad(pred_e, ref)
        return loss 
    
    def calculate_grad(self, pred: np.ndarray, ref: np.ndarray) -> None:
        self.grad = -1 * ref / pred
