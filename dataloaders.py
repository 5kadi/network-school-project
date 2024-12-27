from data import mnist
import numpy as np

x_train, t_train, x_test, t_test = mnist.load()

data_train = x_train.astype(np.float64) / 255
data_test = x_test.astype(np.float64) / 255

def convert_classes(classes: np.ndarray) -> np.ndarray:
    temp = np.empty([classes.shape[0], 10])
    for t in range(classes.shape[0]):
        temp_x = np.zeros([10])
        temp_x[int(classes[t])] = 1
        temp[t] = temp_x
    return temp


classes_train = convert_classes(t_train)
classes_val = convert_classes(t_test)

#print(x_train.shape)






