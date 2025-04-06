import numpy as np
from random import randint
from model import model
from config import WEIGHTS_PATH, NUM_IMAGES
from utils.images import imshow
from data.mnist import get_mnist


model.load_weights(WEIGHTS_PATH)

images, classes = get_mnist() 
rand_idx = randint(0, len(images) - NUM_IMAGES)
sel_images, sel_classes = images[rand_idx:rand_idx + NUM_IMAGES], classes[rand_idx:rand_idx+NUM_IMAGES]

preds = np.zeros(NUM_IMAGES)

for i in range(NUM_IMAGES):
    pred = model(sel_images[i])
    preds[i] = np.argmax(pred)

print(preds == np.argmax(sel_classes, -1))
imshow(sel_images, preds)

