import numpy as np
from scipy import signal as ss

def l2(array1, array2):
    if array1.size > array2.size:
        array1 = ss.resample(array1, array2.size)
    elif array1.size < array2.size:
        array2 = ss.resample(array2, array1.size)
    diff = array1-array2
    return np.linalg.norm(diff, ord =2)

