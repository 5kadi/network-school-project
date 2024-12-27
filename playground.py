from config import *
from network import Network
from loss import CrossEntropyLoss
from data import *
import numpy as np

loss = CrossEntropyLoss()

model = Network(16, 10, 4)
data = np.random.rand(4, 4, 4)
data = data.reshape([-1, 4*4])
actual = np.array(
    [
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
)

for e in range(100):
    for d, r in zip(data, actual):
        y = model(d)
        l = loss(y, r)
        model.backpropagate(loss)
        model.optimize()
        print(f"{e} {l}")


test = model(data[0])
print(np.argmax(test), data.shape)



