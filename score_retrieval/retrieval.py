import numpy as np
from numpy.linalg import norm
from scipy import signal as ss

from score_retrieval.constants import VECTOR_LEN


def resample(arr):
    return ss.resample(arr, VECTOR_LEN)


def L2(arr1, arr2):
    return norm(arr1 - arr2, ord=2)
