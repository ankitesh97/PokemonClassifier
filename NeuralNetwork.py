
import numpy as np
import pandas as pd
import scipy.io as sio
import json


# tanh as activation function
def activate(z):
    return np.tanh(z)

class NeuralNet():
    def __init__(self,n_input,n_output,n_hiddenNodes):
        # define parameters
        # so basically we will have 6 input , n_hiddenNodes and 17 output nodes i.e 3 layer NeuralNet
        self.modelParams = {}
        #total units after adding bias
        n_input += 1
        n_hiddenNodes += 1


def main():
    # data = preprocessData()
    data = sio.loadmat('train.mat')
    X = data['X']
    y = data['y']
