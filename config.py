from pathlib import Path

N_CLASSES = 10
CLASS_NAMES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

EPSILON = 1e-7
LEARNING_RATE = 0.005

EPOCHS = 15
BATCH_SIZE = 50

NUM_IMAGES = 6

WEIGHTS_PATH = Path(__file__).parent.absolute() / "weights.json"