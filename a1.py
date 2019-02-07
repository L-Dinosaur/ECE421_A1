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

    :param W: weight matrix
    :param b: bias
    :param x: feature vector
    :param y: labels
    :param reg: regularization constant
    :return: cross entropy loss of W and b with respect to x and y
    '''
    N = len(x)
    b_vec = np.ones(N) * b
    logit = sigmoid(x.dot(W) + b_vec)
    L = (-1 * y.T[0].dot(np.log(logit)) - (1 - y.T[0]).dot(np.log(1 - logit))) / N + 0.5 * reg * np.dot(W, W)

    return L

def grad(W, b, x, y, reg, lossType):
    if(lossType == "MSE"):
        return gradMSE(W, b, x, y, reg)
    elif(lossType == "CE"):
        return gradCE(W, b, x, y, reg)


def get_loss(W, b, x, y, reg, lossType):
    if(lossType == "MSE"):
        return MSE(W, b, x, y, reg)
    elif(lossType == "CE"):
        return crossEntropyLoss(W, b, x, y, reg)

def gradCE(W, b, x, y, reg):
    '''

    :param W: weight matrix
    :param b: bias
    :param x: feature vector
    :param y: labels
    :param reg: regularization constant
    :return: gradient of CE loss against {W, b}
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
        train_loss[iter_ind] = get_loss(new_W, new_b, x, y, reg, lossType)
        val_loss[iter_ind] = get_loss(new_W, new_b, x_val, y_val, reg, lossType)
        test_loss[iter_ind] = get_loss(new_W, new_b, x_test, y_test, reg, lossType)
        corr, fal = linear_model_eval(new_W, new_b, x_test, y_test)
        accuracy[iter_ind] = corr / (corr + fal)
        iter_ind += 1
        print("Epoch " + str(iter_ind) + " | MSE: " + str(get_loss(new_W, new_b, x, y, reg, lossType)))

    train_loss = train_loss[:iter_ind]
    val_loss = val_loss[:iter_ind]
    test_loss = test_loss[:iter_ind]
    accuracy = accuracy[:iter_ind]

    return new_W, new_b, iter_ind, train_loss, val_loss, test_loss, accuracy


def stochastic_grad_desc(x_train, y_train, x_val, y_val, x_test, y_test, batch_size, epochs, reg_c, lossType_):
    N = len(x_train)
    # , beta1_ = 0.95, beta2_ = 0.99, epsilon_ = 1e-9
    W, b, x, pred, y, loss, optimizer, reg = buildGraph(batch_size, learning_rate=0.001, reg_const=reg_c, lossType=lossType_)
    init = tf.global_variables_initializer()
    indices = np.arange(0, N)
    loss_train = np.zeros(epochs)
    acc_train = np.zeros(epochs)
    loss_val = np.zeros(epochs)
    acc_val = np.zeros(epochs)
    loss_test = np.zeros(epochs)
    acc_test = np.zeros(epochs)
    with tf.Session() as sess:
        sess.run(init)
        for i_epoch in range(epochs):
            avg_loss = 0
            batch_num = int(N / batch_size)
            np.random.shuffle(indices)
            for i_batch in range(batch_num):
                batch_x = x_train[indices[i_batch * batch_size:(i_batch + 1) * batch_size]]
                batch_y = y_train[indices[i_batch * batch_size:(i_batch + 1) * batch_size]]
                batch_y = batch_y.reshape(batch_size, 1).astype("float")
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                w1, b1, x1, pred1, l1 = sess.run([W, b, x, pred, loss], feed_dict={x: batch_x, y: batch_y})
                avg_loss += l1 / batch_num
            print("Epoch: %d | cost: %f" % (i_epoch + 1, avg_loss))

            W_mse, b_mse = sess.run([W, b], feed_dict={x: batch_x, y: batch_y})
            W_mse = W_mse.T[0]
            loss_train[i_epoch] = get_loss(W_mse, b_mse, x_train, y_train, reg_c, lossType_)
            acc_train[i_epoch] = acc(W_mse, b_mse, x_train, y_train, lossType_)
            loss_val[i_epoch] = get_loss(W_mse, b_mse, x_val, y_val, reg_c, lossType_)
            acc_val[i_epoch] = acc(W_mse, b_mse, x_val, y_val, lossType_)
            loss_test[i_epoch] = get_loss(W_mse, b_mse, x_test, y_test, reg_c, lossType_)
            acc_test[i_epoch] = acc(W_mse, b_mse, x_test, y_test, lossType_)

    return loss_train, acc_train, loss_val, acc_val, loss_test, acc_test


def buildGraph(batch_size, beta1_=None, beta2_=None, epsilon_=None, lossType=None, learning_rate=None, reg_const=0.0):
    '''

    :param beta1:
    :param beta2:
    :param epsilon:
    :param lossType:
    :param learning_rate:
    :return:
    '''

    # Initialize weight and bias tensors
    tf.set_random_seed(421)
    # weight and bias tensors
    W = tf.Variable(tf.truncated_normal(shape=(784, 1), stddev=0.5), dtype=tf.float32, name="W")
    b = tf.Variable(0.0, dtype=tf.float32, name="b")
    # Note b is a vector of length N
    # variable tensors
    x = tf.placeholder(shape=(batch_size, 784), dtype=tf.float32, name="x")
    y = tf.placeholder(shape=(batch_size, 1), dtype=tf.float32, name="y")
    reg = tf.constant(reg_const, tf.float32)
    # loss tensor
    if lossType == "MSE":
        pred = tf.matmul(x, W) + b
        err = pred - y
        loss = tf.add(0.5 * tf.reduce_mean(tf.square(err)), 0.5 * reg * tf.reduce_sum(tf.square(W)), name="loss")
    elif lossType == "CE":
        pred = tf.sigmoid(tf.matmul(x, W) + b)
        loss = (-1 * tf.matmul(tf.transpose(y), tf.log(pred + 1e-7)) + tf.matmul(tf.transpose(tf.subtract(y, 1.0)), tf.log(tf.subtract(1.0, pred) + 1e-7))) / batch_size + 0.5 * reg * tf.reduce_sum(tf.square(W))
    # optimizer
    # , beta1=beta1_, beta2=beta2_, epsilon=epsilon_
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-4).minimize(loss)

    return W, b, x, pred, y, loss, optimizer, reg


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


def acc(W, b, x, y, lossType):
    if lossType == "MSE":
        corr, fal = linear_model_eval(W, b, x, y)
    elif lossType == "CE":
        corr, fal = log_model_eval(W, b, x, y)
    return corr / (corr + fal)

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
    plt.plot(ep, train, ep, val, ep, test)
    plt.legend(["training loss", "validation loss", "test_loss"])
    plt.savefig("loss_plot.png")
    plt.clf()


def plot_accuracy(epochs_run, train, val, test):
    ep = np.arange(0, epochs_run)
    plt.plot(ep, train, ep, val, ep, test)
    plt.legend(["training", "validation", "test"])
    plt.savefig("accuracy.png")
    plt.clf()


def plot_accuracy(epoch, train):
    ep = np.arange(0, epoch)
    plt.plot(ep, train)
    plt.legend(["training", "validation", "test"])
    plt.savefig("accuracy.png")
    plt.clf()


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
    runtype = "batch"
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


    if(runtype == "normal"):
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

    if(runtype == "batch"):
        epochs = 5000
        reg = 0
        alpha = 0.005

        print("Initialization finished, start gradient descent")
        start = time.time()
        W, b, epochs_run, train_loss, val_loss, test_loss, iterative_accuracy = \
            grad_descent(W0, b0, x_train, y_train, x_valid, y_valid, x_test, y_test, alpha, epochs, reg, lossType="MSE")
        runtime = time.time() - start
        plot_loss(epochs_run, train_loss, val_loss, test_loss)
        plot_accuracy(epochs_run, iterative_accuracy)
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

    if(runtype == "stochastic"):
        epochs = 700
        batch_size = 500
        reg = 0.0
        start = time.time()
        mse_train, acc_train, mse_val, acc_val, mse_test, acc_test = \
            stochastic_grad_desc(x_train, y_train, x_valid, y_valid, x_test, y_test, batch_size, epochs, reg, "MSE")
        t = time.time() - start

        plot_loss(epochs, mse_train, mse_val, mse_test)
        plot_accuracy(epochs, acc_train, acc_val, acc_test)
        with open("sgd.txt", "w+") as f:
            f.write("SGD Batch Size\n")
            f.write("Final Classification Accuracy: \n")
            f.write("Training %f | Validation %f | Testing %f" % (acc_train[-1], acc_val[-1], acc_test[-1]))

        print("ran in %s seconds" % t)


