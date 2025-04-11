import numpy as np
from data.mnist import get_mnist
from model import *
from config import EPOCHS, WEIGHTS_PATH, BATCH_SIZE

images, classes = get_mnist(shuffle=True, batch_size=BATCH_SIZE)

if WEIGHTS_PATH.is_file():
    model.load_weights(WEIGHTS_PATH)


for e in range(EPOCHS):
    running_corrects = 0
    running_loss = 0.0
    for imgs, refs in zip(images, classes):
        preds = model(imgs)
        loss_val = loss_fn(preds, refs)
        model.backpropagate(loss_fn.grad)

        pred_classes = np.argmax(preds, -1)
        ref_classes = np.argmax(refs, -1)
        
        running_corrects += np.sum(pred_classes == ref_classes)
        running_loss += np.mean(loss_val, 0)
    epoch_acc = running_corrects / len(images * BATCH_SIZE) * 100
    epoch_loss = running_loss / len(images * BATCH_SIZE)
    print(e + 1, epoch_acc, epoch_loss, sep="\t")

model.save_weights(WEIGHTS_PATH)