import pathlib

EPOCHS = 15
LEARNING_RATE = 0.0015
BATCH_SIZE = 100
N_CLASSES = 10
CLASS_NAMES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
EPSILON = 1e-7
WEIGHTS_PATH = fr"{pathlib.Path(__file__).parent.absolute()}\weights.json"
