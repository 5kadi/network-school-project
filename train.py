from net.loss import CrossEntropyLoss
from model import model
from data.mnist import get_mnist
from config import EPOCHS, LEARNING_RATE, EPSILON, WEIGHTS_PATH
import numpy as np

#model.load_weights(WEIGHTS_PATH)
loss = CrossEntropyLoss(EPSILON)

images, classes = get_mnist()

for e in range(EPOCHS):
    running_corrects = 0
    running_loss = 0.0
    for img, ref in zip(images, classes):
        y = model(img)
        l = loss(y, ref)
        model.backpropagate(loss)
        model.optimize(LEARNING_RATE)

        pred = np.argmax(y)
        ref = np.argmax(ref)
        running_corrects += (pred == ref)
        running_loss += l
    epoch_acc = running_corrects / len(images) * 100
    epoch_loss = running_loss / len(images)
    print(e + 1, epoch_acc, epoch_loss, sep="\t")

model.save_weights(WEIGHTS_PATH)
