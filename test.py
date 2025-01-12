import numpy as np 
from data.mnist import get_mnist
from model import model
from config import WEIGHTS_PATH
from utils import imshow 

model.load_weights(WEIGHTS_PATH)

images, classes = get_mnist() 

preds = [0] * 6
data = images[10000:10006]
for i in range(6):
    out = model(images[i])
    preds[i] = np.argmax(out)

imshow(data, preds)







