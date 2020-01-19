import numpy as np

def argmax_top_k(nparray, k):
    return nparray.argsort()[:,-k:][:,::-1]