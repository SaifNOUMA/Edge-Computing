import numpy as np
import random

def compute_entropy(y_pred):
    log_y_pred = np.ma.log(y_pred).filled(0)
    entropies = -np.sum(np.multiply(y_pred,log_y_pred), axis=-1)
    return entropies
    