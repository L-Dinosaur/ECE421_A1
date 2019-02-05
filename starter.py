import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def loadData():
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

def MSE(W, b, x, y, reg):
    '''
    :param W: Weight matrix
    :param b: bias matrix
    :param x: data
    :param y: labels
    :param reg: regularization param
    :return: mean square error loss
    '''
    N = len(x)
    Ld = 0
    for n in range(N):
        Ld += np.square(np.dot(W, x[n]) + b - y[n])
    L = Ld / (2 * N) + reg * np.dot(W, W) / 2
    return L

def gradMSE(W, b, x, y, reg):
    N = len(x)
    d = len(W)
    b_grad = 0
    w_grad = np.zeros(d)
    for n in range(N):
        # Calculate {w' dot x(n) + b - y(n)}
        error = np.dot(W, x[n]) + b - y[n]
        # Calculate b_grad
        b_grad += error
        # Calculate each individual component of w_grad
        for k in range(d):
            w_grad[k] += x[n][k] * error + reg * W[k]

    b_grad = b_grad / N
    for i in range(d):
        w_grad[k] = w_grad[k] / N

    return w_grad, b_grad


def crossEntropyLoss(W, b, x, y, reg):
    # Your implementation here
    pass

def gradCE(W, b, x, y, reg):
    # Your implementation here
    pass

def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
    # Your implementation here
    pass

def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    # Your implementation here
    pass


if __name__ == '__main__':
    df = loadData()
    print("done")
