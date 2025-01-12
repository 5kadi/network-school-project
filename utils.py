import matplotlib.pyplot as plt 
import numpy as np
from config import CLASS_NAMES


def imshow(samples: np.ndarray, classes: np.ndarray, amount: int = 6) -> None:
    for i in range(amount):
        plt.subplot(2, 3, i + 1)
        img = samples[i].reshape([28, 28])
        plt.imshow(img, cmap="gray")
        plt.title(CLASS_NAMES[int(classes[i])])
    plt.show()                                                                                                            