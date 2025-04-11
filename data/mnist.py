import numpy as np 
from pathlib import Path
from config import N_CLASSES

def get_mnist(shuffle: bool = False, batch_size: int = None) -> tuple[np.ndarray, np.ndarray]:
    with np.load(Path(__file__).parent.absolute() / "mnist.npz") as f:
        images, classes = f["x_train"], f["y_train"]
    images = images.astype("float64") / 255
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    classes = np.eye(N_CLASSES)[classes]

    if shuffle:
        p = np.random.permutation(len(images))
        images = images[p]
        classes = classes[p]

    if batch_size:
        images = np.split(images, len(images) / batch_size)
        classes = np.split(classes, len(classes) / batch_size)

    return images, classes  