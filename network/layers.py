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
        X_expanded = np.expand_dims(X, -2)
        self.derivatives_to_weights = np.repeat(X_expanded, self.outputs, -2)
    
    def backpropagate(self, next_grad: np.ndarray) -> np.ndarray:
        delta_weights_T = np.expand_dims(next_grad, -1) * self.derivatives_to_weights
        delta_weights = np.moveaxis(delta_weights_T, -1, -2)
        delta_weights = np.mean(delta_weights, 0)
        self.optimize_weights(delta_weights)
        
        grad_by_prev = (np.expand_dims(next_grad, -2) @ self.derivatives_to_prev).squeeze()
        return grad_by_prev
    
    def optimize_weights(self, delta_weights: np.ndarray) -> None:
        self.weights -= LEARNING_RATE * delta_weights

class SoftmaxLayer(AbstractLayer):
    def __call__(self, X: np.ndarray) -> np.ndarray:
        z = X - np.max(X)
        signal = np.exp(z) / np.sum(np.exp(z), -1, keepdims=True) 
        self.calculate_derivatives(signal)
        return signal
    
    def calculate_derivatives(self, signal: np.ndarray) -> None:
        diagonal_matrix = np.expand_dims(signal, -2) * np.eye(signal.shape[-1])
        relations_matrix = np.expand_dims(signal, -1) * np.expand_dims(signal, -2)
        self.derivatives_to_prev = diagonal_matrix - relations_matrix

    def backpropagate(self, next_grad: np.ndarray) -> np.ndarray:
        grad = (np.expand_dims(next_grad, -2) @ self.derivatives_to_prev).squeeze()
        return grad