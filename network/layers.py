import numpy as np
from abc import ABC, abstractmethod
from config import LEARNING_RATE

class AbstractLayer(ABC):
    @abstractmethod
    def __call__(self, X: np.ndarray) -> np.ndarray: ...
    @abstractmethod
    def calculate_derivatives(self, *args, **kwargs) -> None: ...
    @abstractmethod
    def backpropagate(self, next_grad: np.ndarray) -> np.ndarray: ...

class LinearLayer(AbstractLayer):
    def __init__(self, inputs: int, outputs: int) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.weights = np.random.uniform(0, 1 / np.sqrt(self.inputs), [self.inputs, self.outputs])

    def __call__(self, X: np.ndarray) -> np.ndarray:
        signal = X @ self.weights
        self.calculate_derivatives(X)
        return signal
    
    def calculate_derivatives(self, X: np.ndarray) -> None:
        self.derivatives_to_prev = self.weights.T
        self.derivatives_to_weights = np.repeat(X[np.newaxis, :], self.outputs, 0)
    
    def backpropagate(self, next_grad: np.ndarray) -> np.ndarray:
        delta_weights = (next_grad[:, np.newaxis] * self.derivatives_to_weights).T
        self.optimize_weights(delta_weights)
        grad_by_prev = next_grad @ self.derivatives_to_prev
        return grad_by_prev
    
    def optimize_weights(self, delta_weights: np.ndarray) -> None:
        self.weights -= LEARNING_RATE * delta_weights

class SoftmaxLayer(AbstractLayer):
    def __call__(self, X: np.ndarray) -> np.ndarray:
        z = X - np.max(X)
        signal = np.exp(z) / np.sum(np.exp(z)) 
        self.calculate_derivatives(signal)
        return signal
    
    def calculate_derivatives(self, signal: np.ndarray) -> None:
        self.derivatives_to_prev = np.diagflat(signal) - signal[:, np.newaxis] * signal[np.newaxis, :]

    def backpropagate(self, next_grad: np.ndarray) -> np.ndarray:
        grad = next_grad @ self.derivatives_to_prev
        return grad