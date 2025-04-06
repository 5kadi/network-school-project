import numpy as np
import json
from pathlib import Path
from .layers import AbstractLayer

class Network:
    def __init__(self, layers: list[AbstractLayer]) -> None:
        self.layers = layers

    def __call__(self, X: np.ndarray) -> np.ndarray:
        x = X
        for layer in self.layers:
            x = layer(x)
        return x
    
    def backpropagate(self, grad: np.ndarray) -> None:
        for layer in reversed(self.layers):
            grad = layer.backpropagate(grad)

    def save_weights(self, path: str | Path) -> None:
        weights_list = [layer.weights.tolist() for layer in self.layers if hasattr(layer, "weights")]
        with open(path, "w") as f: 
            json.dump(weights_list, f)

    def load_weights(self, path: str | Path) -> None:
        with open(path, "r") as f:
            weights_list = json.load(f)
        for i in range(len(weights_list)):
            layer = self.layers[i]
            if hasattr(layer, "weights"):
                layer.weights = np.array(weights_list[i])