
import numpy as np
import pandas as pd
import scipy.io as sio
import json

N_CLASSES = 17

# tanh as activation function
def activate(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_grad(z):
    sigValue = activate(z)
    return np.multiply(sigValue,(1-sigValue))

def _softmax(z):
    '''calculates the probabilities'''
    e_x = np.exp(z-np.max(z))
    return e_x/e_x.sum()

np.random.seed(0)

class NeuralNet():
    def __init__(self,n_input,n_output,n_hiddenNodes=16):
        # define parameters
        # so basically we will have 6 input , n_hiddenNodes and 17 output nodes i.e 3 layer NeuralNet
        self.n_input = n_input
        self.n_output = n_output
        self.n_hiddenNodes = n_hiddenNodes
        #so base structure would be 6x11x17
        #total units after adding bias
        #layer 1 theta
        # theta1 = np.radom.randn(n_hiddenNodes,n_input+1)/np.sqrt(n_input)
        #layer 2 theta
        # theta2 = np.random.randn(n_output,n_hiddenNodes+1)/np.sqrt(n_hiddenNodes)
        # total dimensions i.e layer 1 + layer 2
        totalSize = n_hiddenNodes*(n_input+1) + n_output*(n_hiddenNodes+1)
        self.theta = (np.random.randn(size = totalSize)-0.5)*0.25


    # def forward propogation
    def _forward_prop(self,theta1,theta2,X):
        #1st layer values
        z1 = np.dot(X,theta1.T)
        #activate those
        a1 = activate(z1)
        #append the bias unit to the first layer
        ones = np.ones((X.shape[0],1))
        a1 = np.hstack((ones,a1))
        #calculate the layer two values
        z2 = np.dot(a1,theta2.T)
        #APPLY THE softmax functionto calculate probabilities
        a2 = _softmax(z2)
        return z1,a1,z2,a2

    def _makeY(self,y):
        y_new = np.array([0 for i in range(N_CLASSES)])
        y_new[y-2] = 1
        return y_new

    def cost(self,theta,X,y):
        #theta contains all theta of layer 1 and layer 2
        theta1Total = (self.n_hiddenNodes)*(self.n_input+1)
        theta2Total = (self.n_output)*(self.n_hiddenNodes+1)
        theta1 = np.reshape(theta[:theta1Total],(self.n_hiddenNodes,self.n_input+1))
        theta1 = np.reshape(theta[:theta2Total],(self.n_output,self.n_hiddenNodes+1))
        #calculate using feed forward pass
        z1,a1,z2,a2 = self._forward_prop(theta1,theta2,X)
        # basically output for m training examples have been outputed
        m = a2.shape[0]
        # basically for all training examples
        J = 0
        for i in range(0,m):
            yCurr = self._makeY(y[i][0])
            first_term = np.multiply(-yCurr,np.log(a2[i]))
            second_term = np.multiply(1-yCurr,np.log(1-a2[i]))
            J += np.sum(first_term-second_term)
        J /= m
        return J

    def backprop(self,theta,X,y):
        #for every training example perform feed forward and calculate values(this is done in a vector format)

        theta1Total = (self.n_hiddenNodes)*(self.n_input+1)
        theta2Total = (self.n_output)*(self.n_hiddenNodes+1)
        theta1 = np.reshape(theta[:theta1Total],(self.n_hiddenNodes,self.n_input+1))
        theta1 = np.reshape(theta[:theta2Total],(self.n_output,self.n_hiddenNodes+1))
        z1,a1,z2,a2 = self._forward_prop(theta1,theta2,X)
        #now for every training example perform backpropogation and add to the error to calculate gradients
        m = X.shape[0]
        changeMat = np.zeros(theta.shape)
        for i in range(0,m):
            yCurr = self._makeY(y[i][0])
            d3t = a2[i] - yCurr
            #make the delta a column vector
            d3t = d3t[np.newaxis].T
            #now take the theta transpose and multiply it for the error in the second layer
            derivative = sigmoid_grad(z1[i])[np.newaxis].T
            d2t = np.multiply(np.dot(theta2.T,d3t),derivative)


    def fit(self,X,y):
        #to fit the models
        # append the bias unit
        ones = np.ones((X.shape(0),1))
        X = np.hstack((ones,X))


def main():
    # data = preprocessData()
    data = sio.loadmat('train.mat')
    X = data['X']
    y = data['y']

if __name__ == '__main__':
    main()
