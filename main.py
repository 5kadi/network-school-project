

import numpy as np 
"""
X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([i*2 for i in X], dtype=np.float32)

w = 0.0 
lr = 0.1

def forward(x):
    return w * x

def loss(pred, ref):
    loss = ((pred - ref)**2).mean()
    return loss 

def gradient(x, y, wx):
    return np.dot(x*2, wx - y)/x.size 

EPOCHS = 25 

for i in range(EPOCHS):
    wx = forward(X)
    l = loss(wx, Y)
    grad = gradient(X, Y, wx)
    w -= lr * grad 
    print(l, grad, sep="\n")

print(forward(X))
"""
a = np.array(
    [
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [4, 1, 2, 3],
        [4, 2, 1, 3]
    ]
)
b = np.array(
    [
       [0, 1, 100, 1000],
       [0, 0, 0, 1],
       [0, 0, 0, 0],
       [10, 10, 10, 10]
    ]
)
#b = np.expand_dims(0)
actual = np.array(
    [
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
)

a = a[:, :, np.newaxis] * a[:, np.newaxis, :]
b = a.mean(0)

print(b, sep="\n")
