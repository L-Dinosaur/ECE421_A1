import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

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
    b_vec = np.ones(N) * b
    error = x.dot(W) + b_vec - y.T[0]
    L = 0.5 * np.dot(error, error) / N + 0.5 * reg * np.dot(W, W)

    return L


def gradMSE(W, b, x, y, reg):
    '''
    :param W: Weight matrix
    :param b: bias matrix
    :param x: data matrix
    :param y: labels
    :param reg: regularization parameter
    :return: gradient of MSE against {weights, bias}
    '''

    N = len(x)
    d = len(W)

    g_w = 0
    g_b = (x.dot(W) - y.T).sum() / N + b
    g_w = x.T.dot((x.dot(W) - y.T + b).T).T/N + reg*W

    return g_w[0], g_b


def sigmoid(z):
    return 1 / (1 + np.exp(-1 * z))

def crossEntropyLoss(W, b, x, y, reg):
    '''

    :param W:
    :param b:
    :param x:
    :param y:
    :param reg:
    :return:
    '''
    N = len(x)
    b_vec = np.ones(N) * b
    logit = sigmoid(x.dot(W) + b_vec)
    L = (-1 * y.T[0].dot(np.log(logit)) - (1 - y.T[0]).dot(np.log(1 - logit))) / N + 0.5 * reg * np.dot(W, W)

    return L

def grad(W, b, x, y, reg, lossType):
    if(lossType == "linear"):
        return gradMSE(W, b, x, y, reg)
    elif(lossType == "log"):
        return gradCE(W, b, x, y, reg)


def loss(W, b, x, y, reg, lossType):
    if(lossType == "linear"):
        return MSE(W, b, x, y, reg)
    elif(lossType == "log"):
        return crossEntropyLoss(W, b, x, y, reg)

def gradCE(W, b, x, y, reg):
    '''

    :param W:
    :param b:
    :param x:
    :param y:
    :param reg:
    :return:
    '''
    N = len(x)
    b_vec = np.ones(N) * b
    g_b = (sigmoid(x.dot(W) + b_vec) - y.T[0]).sum() / N
    g_w = x.T.dot(sigmoid(x.dot(W) + b) - y.T[0]) / N + reg * W
    return g_w, g_b


def grad_descent(W, b, x, y, x_val, y_val, x_test, y_test, alpha, epochs, reg, error_tol=1e-7, lossType="None"):

    '''

    :param W: Weight matrix
    :param b: bias matrix
    :param x: trainingData
    :param y: trainingLabels
    :param alpha: learning rate
    :param epochs: number of passes through training data
    :param reg: regularization constant
    :param error_tol: error tolerance
    :return: learned weights and bias

    '''


    w_grad, b_grad = grad(W, b, x, y, reg, lossType)

    new_W = W - alpha * w_grad
    new_b = b - alpha * b_grad
    iter_ind = 0
    train_loss = np.zeros(epochs)
    val_loss = np.zeros(epochs)
    test_loss = np.zeros(epochs)
    accuracy = np.zeros(epochs)
    while((np.linalg.norm(np.append(W, b) - np.append(new_W, new_b)) > error_tol) and iter_ind < epochs):
        W = new_W
        b = new_b
        w_grad, b_grad = grad(W, b, x, y, reg, lossType)
        new_W = W - alpha * w_grad
        new_b = b - alpha * b_grad

        # Calculate losses
        train_loss[iter_ind] = loss(new_W, new_b, x, y, reg, lossType)
        val_loss[iter_ind] = loss(new_W, new_b, x_val, y_val, reg, lossType)
        test_loss[iter_ind] = loss(new_W, new_b, x_test, y_test, reg, lossType)
        corr, fal = linear_model_eval(new_W, new_b, x_test, y_test)
        accuracy[iter_ind] = corr / (corr + fal)
        iter_ind += 1
        print("Epoch " + str(iter_ind) + " | MSE: " + str(loss(new_W, new_b, x, y, reg, lossType)))

    train_loss = train_loss[:iter_ind]
    val_loss = val_loss[:iter_ind]
    test_loss = test_loss[:iter_ind]
    accuracy = accuracy[:iter_ind]

    return new_W, new_b, iter_ind, train_loss, val_loss, test_loss, accuracy

def normal_equation(x, y):
    W = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y.T[0])
    b = -1 * (x.dot(W) - y.T).sum() / N
    return W, b

def log_model_eval(W, b, x, y):
    '''

    :param W: weight matrix
    :param b: bias matrix
    :param x: test data
    :param y: labels
    :return:

    '''

    N = len(x)
    b_vec = np.ones(N) * b

    y_hat = sigmoid(x.dot(W) + b_vec)
    error = np.round(np.abs(y_hat - y.T[0]))
    fal = error.sum()
    corr = N - fal

    return corr, fal

def linear_model_eval(W, b, x, y):
    '''

    :param W: weight matrix
    :param b: bias matrix
    :param x: test data
    :param y: labels
    :return:

    '''

    N = len(x)
    b_vec = np.zeros(N) * b
    corr = 0
    fal = 0

    y_hat = x.dot(W) + b_vec
    error = y_hat - y.T[0]

    for n in range(N):
        if(y_hat[n] >= 0.5):
            y_hat[n] = 1
        else:
            y_hat[n] = 0
        if(y_hat[n] == y[n]):
            corr += 1
        else:
            fal += 1
    return corr, fal


def plot_loss(epochs_run, train, val, test):
    ep = np.arange(0, epochs_run)
    plt.plot(ep, train_loss, ep, val_loss, ep, test_loss)
    plt.legend(["training loss", "validation loss", "test_loss"])
    plt.savefig("loss_plot.png")
    plt.clf()


def plot_accuracy(acc):
    ep = np.arange(0, epochs_run)
    plt.plot(ep, acc)
    plt.legend(["test set accuracy"])
    plt.savefig("accuracy.png")
    plt.clf()

def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    '''

    :param beta1:
    :param beta2:
    :param epsilon:
    :param lossType:
    :param learning_rate:
    :return:
    '''
    pass

if __name__ == '__main__':

    print("Regression Example")
    X_train, X_valid, X_test, y_train, y_valid, y_test = loadData()
    X0 = X_train[0]
    d = len(X0) * len(X0[0])
    N = len(X_train)
    Nv = len(X_valid)
    Nt = len(X_test)
    W0 = np.ones(d) / 1000
    b0 = 1
    normal = False
    # Flatten the input images into vectors
    x_train = np.zeros((N, d))
    x_valid = np.zeros((Nv, d))
    x_test = np.zeros((Nt, d))
    for n, xn in enumerate(X_train):
        x_train[n] = np.reshape(xn, d)
    for n, xn in enumerate(X_valid):
        x_valid[n] = np.reshape(xn, d)
    for n, xn in enumerate(X_test):
        x_test[n] = np.reshape(xn, d)

    if(normal):
        start = time.time()
        Wn, bn = normal_equation(x_train, y_train)
        runtime = time.time() - start
        corr, fal = linear_model_eval(Wn, bn, x_test, y_test)
        accuracy = corr / (corr + fal)
        fin_tr_loss = MSE(Wn, bn, x_train, y_train, 0)
        with open("normal.txt", "w+") as f:
            f.write("Normal Equation\n")
            f.write("Runtime: %s seconds\n" % runtime)
            f.write("Final Training Loss: %f \n" % fin_tr_loss)
            f.write("Accuracy: %f" % accuracy)


    epochs = 5000
    reg = 0
    alpha = 0.005

    print("Initialization finished, start gradient descent")
    start = time.time()
    W, b, epochs_run, train_loss, val_loss, test_loss, iterative_accuracy = \
        grad_descent(W0, b0, x_train, y_train, x_valid, y_valid, x_test, y_test, alpha, epochs, reg, lossType="linear")
    runtime = time.time() - start
    plot_loss(epochs_run, train_loss, val_loss, test_loss)
    plot_accuracy(iterative_accuracy)
    corr, fal = linear_model_eval(W, b, x_test, y_test)
    accuracy = corr / (corr + fal)
    with open("run1.txt", "w+") as f:
        f.write("learning rate = 0.05\n")
        f.write("Runtime: %s seconds\n" % runtime)
        f.write("Epochs run: %d \n" % epochs_run)
        f.write("Accuracy: %f\n" % accuracy)
        f.write("Final Training Loss: %f \n" % train_loss[-1])
        f.write("Final Validation Loss: %f \n" % val_loss[-1])
        f.write("Final Testing Loss: %f \n" % test_loss[-1])

    print("--- %s seconds ---" % runtime)
    print("--- correct classification: " + str(corr))
    print("--- false classification:   " + str(fal))
    print("--- accuracy:               " + str(corr / (corr + fal)))
