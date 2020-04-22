import numpy as np


#dummy functions to check plot correctness
first = True

def plot_iter_codebook():
    if first==False:
        return np.load('codebook.npy')
    iters = np.arange(10,110,10)
    loss = np.array([20000,22000,19000,18000,18020,17000,15000,14000,13000,12000])
    return iters,loss


def plot_data_vs_testing():
    if first==False:
        return np.load('results.npy')
    size = np.arange(200,2000,200)
    svm = [20000,20000,18080,19000,18000,18050,16000,15000,14000]
    ssvm = [12000,13000,11020,8000,7000,8500,6500,3000,4500]
    lssvm = [12020,12000,11030,8060,9000,9500,6500,2000,3500]
    return size,svm,ssvm,lssvm
