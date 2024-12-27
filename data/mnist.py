import numpy as np
from urllib import request
import gzip
import pickle

filename = [
    ["training_images", "MNIST/raw/train-images-idx3-ubyte.gz"],
    ["test_images", "MNIST/raw/t10k-images-idx3-ubyte.gz"],
    ["training_labels", "MNIST/raw/train-labels-idx1-ubyte.gz"],
    ["test_labels", "MNIST/raw/t10k-labels-idx1-ubyte.gz"]
]


def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")

def init():
    save_mnist()

def load():
    with open("data\mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

if __name__ == '__main__':
    init()