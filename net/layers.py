import numpy as np

class BaseLayer:
    def __init__(self, inputs: int, outputs: int) -> None: ...
    def __call__(self, X: np.ndarray) -> np.ndarray: ...
    def calculate_grads(self, *args) -> None: ...
    def backpropagate(self, next_grads: np.ndarray) -> np.ndarray: ...
    def optimize(self, learning_rate: float) -> None: ...

class LinearLayer(BaseLayer):
    def __init__(self, inputs: int, outputs: int) -> None:
        self.inputs = inputs
        self.outputs = outputs 
        # w11 w21 w31  x - inputs (i)
        # w12 w22 w32  y - outputs (j)
        # w13 w23 w33
        self.weights = np.random.uniform(0, 1 / np.sqrt(self.inputs), [self.outputs, self.inputs])

    def __call__(self, X: np.ndarray) -> np.ndarray: 
        #надо транспонировать, потому что y весов должен быть равен x входных данных
        self.signal = X @ self.weights.transpose()
        self.calculate_grads(X)
        return self.signal
    
    def calculate_grads(self, o_input: np.ndarray) -> None:
        #doj/dwij = oi
        # do1/dw11 do1/dw21 x - wij (o-1, inputs)
        # do2/dw12 do2/dw22 y - o (outputs)
        #сводится к следующему:
        # (o-1)1 (o-1)2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
        # (o-1)1 (o-1)2
        self.grads = np.repeat(o_input[np.newaxis, :], self.outputs, 0)

    def backpropagate(self, next_grads: np.ndarray) -> np.ndarray:
        #doj/doi = wij
        # do1/d(o-1)1 do1/d(o-1)2   x - (o-1), inputs
        # do2/d(o-1)1 do2/d(o-1)2   y - o, outputs
        #сводится к следующему:
        # w11 w21 , т.е. просто матрица весов
        # w12 w22
        #тогда (dE/doi = SUMj dE/doj * doj/doi (общий случай)):
        # dE/do1 dE/do2  X  w11 w21  =  dE/d(o-1) dE/d(o-2)
        #                   w12 w22
        prev_dE = next_grads @ self.weights
        #транспонировано -> dE/do1  *  do1/dw11 do1/dw21  =  dE/dw11 dE/dw21  (нужно именно такое умножение)
        #                   dE/do2     do2/dw12 do1/dw22     dE/dw12 dE/dw22
        self.delta_w = next_grads[:, np.newaxis] * self.grads 
        return prev_dE
    
    def optimize(self, learning_rate: float) -> None: 
        #w = w - lr * delta_w
        self.weights -= learning_rate * self.delta_w 
    
class SoftmaxLayer(BaseLayer):
    def __init__(self, inputs: int = None, outputs: int = None) -> None:
        self.weights = np.array([])

    def __call__(self, X: np.ndarray) -> np.ndarray:
        #предотвращает перегрузку значений экспоненты
        z = X - np.max(X)
        #si = e**oi / SUMj e**oj (доля связи)
        self.signal = np.exp(z) / np.sum(np.exp(z), -1) 
        self.calculate_grads()
        return self.signal
    
    def calculate_grads(self) -> None:
        #dsi/doi = si(1 - si) (i of s == i of oi)
        #dsi/doI = -1 * si * sI (i of si != I of oI)
        # ds1/do1 ds1/do2   x - o 
        # ds2/do1 ds2/do2   y - s
        #но сводится к следующему:
        # s1-s1**2 -s1s2     , т.е.  s1 0  -  s1**2 s1s2    x - s <- (2 матрица)
        # -s2s1    s2-s2**2          0 s2     s2s1  s2**2   y - s
        # x - s, y -s, поэтому транспонирование ничего не меняет
        self.grad = np.diagflat(self.signal) - self.signal[:, np.newaxis] * self.signal[np.newaxis, :]

    def backpropagate(self, next_grads: np.ndarray) -> np.ndarray:
        #dE/doi = SUMj dE/doj * doj/doi (общий случай)
        # dE/ds1 dE/ds2  X  ds1/do1 ds1/do2  =  dE/do1 dE/do2
        #                   ds2/do1 ds2/do2
        prev_dE = next_grads @ self.grad
        return prev_dE
    
    def optimize(self, learning_rate: float) -> None:
        pass

"""
class ReLU:    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = X * (X > 0) 
        #X > 0 = True => X * 1 = X
        #X <= 0 => X > 0 = False => X * 0 = 0
        #print(X)
        return X
"""   



