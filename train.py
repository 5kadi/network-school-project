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

for e in range(5):
    running_corrects = 0
    running_loss = 0.0
    for img, ref in zip(data_train[:100], classes_train[:100]):
        #print(img.shape)
        y = model(img)
        l = loss(y, ref)
        model.backpropagate(loss)
        model.optimize()

        pred = np.argmax(y)
        ref = np.argmax(ref)
        running_corrects += (pred == ref)
        print(l)
        running_loss += l
    epoch_acc = running_corrects / len(data_train[:100]) * 100
    epoch_loss = running_loss / len(data_train[:100])
    print(e, epoch_acc, epoch_loss, sep="\t")


preds = [0] * 6
data = data_test[:6]
for i in range(6):
    out = model(data_test[i])
    preds[i] = np.argmax(out)

imshow(data, preds)