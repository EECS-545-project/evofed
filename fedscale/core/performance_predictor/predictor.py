from curve_model import curving_model
import numpy as np

def predict(y_train: list, endingpoint: int) -> int:
    y_train = np.array(y_train)
    curving = curving_model(y_train, endingpoint)
    return curving.get_expected_value()[0]