from network.network import Network
from network.layers import *
from network.loss import CrossEntropyLoss

model = Network(
    [
        LinearLayer(784, 100),
        LinearLayer(100, 10),
        SoftmaxLayer()
    ]
)
loss_fn = CrossEntropyLoss()
