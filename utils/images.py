import matplotlib.pyplot as plt 
import numpy as np
from config import CLASS_NAMES

def imshow(images: np.ndarray, classes: np.ndarray | list, amount: int = 6) -> None:
    for i in range(amount):
        plt.subplot(2, 3, i + 1)
        img = images[i].reshape([28, 28])
        plt.imshow(img, cmap="gray")
        plt.title(CLASS_NAMES[int(classes[i])])
    plt.show()   