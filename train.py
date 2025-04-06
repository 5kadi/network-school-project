import numpy as np
from data.mnist import get_mnist
from model import *
from config import EPOCHS, WEIGHTS_PATH

images, classes = get_mnist(shuffle=True)

if WEIGHTS_PATH.is_file():
    model.load_weights(WEIGHTS_PATH)

for e in range(EPOCHS):
    running_corrects = 0
    running_loss = 0.0
    for img, ref in zip(images, classes):
        pred = model(img)
        loss_val = loss_fn(pred, ref)
        model.backpropagate(loss_fn.grad)

        pred_class = np.argmax(pred)
        ref_class = np.argmax(ref)
        
        running_corrects += (pred_class == ref_class)
        running_loss += loss_val
    epoch_acc = running_corrects / len(images) * 100
    epoch_loss = running_loss / len(images)
    print(e + 1, epoch_acc, epoch_loss, sep="\t")

model.save_weights(WEIGHTS_PATH)