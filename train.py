from loss import CrossEntropyLoss
from network import Network
from utils import imshow 
from dataloaders import *
from config import EPOCHS

#data = np.random.rand(4, 4)
#data = data.reshape([4*4])
#actual = np.array([1, 0, 0, 0])

model = Network(784, 100, 10)
loss = CrossEntropyLoss()

for e in range(100):
    running_corrects = 0
    running_loss = 0.0
    for i in range(100):
        #print(img.shape)
        y = model(data_train[i])
        l = loss(y, classes_train[i])
        model.backpropagate(loss)
        model.optimize()

        pred = np.argmax(y)
        ref = np.argmax(classes_train[i])
        running_corrects += (pred == ref)
        running_loss += l
    epoch_acc = running_corrects / len(data_train[0]) * 100
    epoch_loss = running_loss / len(data_train[0])
    print(e, epoch_acc, epoch_loss, sep="\t")


preds = [0] * 6
data = data_train[:1, :]
for i in range(len(data)):
    out = model(data[i])
    preds[i] = np.argmax(out)
    print(preds[i])

print(classes_train[0])
imshow(data, preds, 1)
