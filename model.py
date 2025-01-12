from net.layers import LinearLayer, SoftmaxLayer
from net.network import BaseNetwork
from config import CLASS_NAMES

class Network_2L1S(BaseNetwork):
    def __init__(self, inp: int, hid: int, out: int) -> None:
        self.layers = [
            LinearLayer(inp, hid),
            LinearLayer(hid, out),
            SoftmaxLayer()
        ]

model = Network_2L1S(784, 100, len(CLASS_NAMES))