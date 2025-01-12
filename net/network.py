import numpy as np
from .loss import BaseLoss
import json

class BaseNetwork:
    def __init__(self, inp: int, hid: int, out: int) -> None: ...

    def __call__(self, X: np.ndarray) -> np.ndarray:
        x = X
        for layer in self.layers:
            x = layer(x)
        return x
    
    def backpropagate(self, loss: BaseLoss) -> None:
        grad = loss.grad 
        for layer in reversed(self.layers):
            grad = layer.backpropagate(grad)

    def optimize(self, learning_rate: float) -> None:
        for layer in self.layers:
            layer.optimize(learning_rate)

    def save_weights(self, path: str) -> None:
        weights_dict = dict(enumerate([layer.weights.tolist() for layer in self.layers]))
        with open(path, "w") as f: 
            json.dump(weights_dict, f)

    def load_weights(self, path: str) -> None:
        with open(path, "r") as f:
            weights_dict = json.load(f)
        for i in range(len(weights_dict)):
            self.layers[i].weights = np.array(weights_dict[str(i)])




