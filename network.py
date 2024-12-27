import numpy as np
from loss import CrossEntropyLoss
from config import LEARNING_RATE, EPSILON


class LinearLayer:
    def __init__(self, inputs: int, outputs: int) -> None:
        self.inputs = inputs
        self.outputs = outputs 
        # w11 w21 w31  x - inputs (i)
        # w12 w22 w32  y - outputs (j)
        # w13 w23 w33
        self.weights = np.random.rand(self.outputs, self.inputs)

    def __call__(self, X: np.ndarray) -> np.ndarray: 
        #shape X должен быть [z, y*x], где z - кол-во образцов, y*x = self.inputs
        #weights.shape[1] = X.shape[1], т.к weights.shape[1] = X.shape[1] = inputs
        #надо транспонировать, потому что x весов должен быть равен y входных данных, поэтому размещаем x по y
        self.signal = self.weights @ X.transpose()
        #возвращаем кол-во образцов по y для лучшей совместимости с другими слоями
        self.signal = self.signal.transpose() 
        self.calculate_grads(X)
        #print(self.signal)
        return self.signal
    
    def calculate_grads(self, o_input: np.ndarray) -> None:
        #doj/dwij = oi
        # do1/dw11 do2/dw12 x - o (x - len oj)
        # do1/dw21 do2/dw22 y - w (o-1) (y - len oi,  т.к wij: i = x)
        self.grads = np.empty([self.inputs, self.outputs])
        for oi in range(self.inputs): 
            for oj in range(self.outputs):
                self.grads[oi][oj] = o_input[oi]

    def backpropagate(self, next_grads: np.ndarray) -> np.ndarray:
        #dE/doi = SUMj dE/doj * doj/doi (общий случай)
        #doj/doi = wij
        # dE/do1 dE/do2  X  w11 (do1/do-1) w21 (do1/do-2) 
        #                   w12 (do2/do-1) w22 (do2/do-2) 
        prev_dE = next_grads @ self.weights
        #dE/dwij
        self.deltaW = LEARNING_RATE * (self.grads * next_grads).transpose() #нужно конкретно умножение
        #print(self.deltaW.shape, self.weights.shape, "\t", self.grads.shape, next_grads.shape)
        return prev_dE
    
    def optimize(self):
        self.weights -= self.deltaW

    def __str__(self) -> str:
        return str(self.weights.shape)
    
class Softmax:
    def __call__(self, X: np.ndarray) -> np.ndarray:
        z = X - np.max(X) #предотвращает перегруз значений
        self.signal = np.exp(z) / np.sum(np.exp(z), -1)
        #print(X)
        self.calculate_grads(z, self.signal)
        return self.signal
    
    def calculate_grads(self, o: np.ndarray, s: np.ndarray) -> None:
        #dsi/doi = si(1 - si) (i of s == i of oi)
        #dsi/doI = -1 * si * sI (i of si != I of oI)
        # ds1/do1 ds2/do1 x - s
        # ds1/do2 ds2/do2 y - o
        self.grads = np.empty([o.shape[-1], o.shape[-1]]) 
        for oi in range(o.shape[-1]):
            for si in range(o.shape[-1]):
                if oi == si:
                    self.grads[oi][si] = s[si] * (1 - s[si]) #TODO: next_grad
                else:
                    self.grads[oi][si] = -1 * s[si] * s[oi]

    def backpropagate(self, next_grads: np.ndarray) -> np.ndarray:
        #chain rule
        # ds1/do1 ds2/do1  X  dE/ds1 <- transposed  =  dE/do1
        # ds1/do2 ds2/do2     dE/ds2 <-                dE/do2
        prev_dE = self.grads @ next_grads.transpose() #транспонировать, чтобы x = y
        prev_dE = prev_dE.transpose()
        return prev_dE

class ReLU:    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = X * (X > 0) 
        #X > 0 = True => X * 1 = X
        #X <= 0 => X > 0 = False => X * 0 = 0
        #print(X)
        return X
    
class Network:
    def __init__(self, inp: int, hid: int, out: int) -> None:
        self.layer1 = LinearLayer(inp, hid) 
        self.layer2 = LinearLayer(hid, out)
        self.softmax = Softmax()

    def __call__(self, X: np.ndarray) -> np.ndarray:
        x = self.layer1(X)
        x = self.layer2(x)
        x = self.softmax(x)
        return x
    
    def backpropagate(self, loss: CrossEntropyLoss) -> None:
        softmax_grad = self.softmax.backpropagate(loss.grad)
        layer2_grad = self.layer2.backpropagate(softmax_grad)
        self.layer1.backpropagate(layer2_grad)

    def optimize(self) -> None:
        self.layer2.optimize()
        self.layer1.optimize()


model  = Network(16, 10, 4)



