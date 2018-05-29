import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = np.genfromtxt('dataset.csv', delimiter = ',', skip_header = 1)
x = data[:, :2]
y = data[:, 2]

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.1, stratify = y)
X_train = X_train.T
X_test = X_test.T
Y_train = Y_train.reshape(1, Y_train.shape[0])
Y_test = Y_test.reshape(1, Y_test.shape[0])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def init_param(dim):
    w = np.random.rand(dim, 1)
    b = 0
    return w, b

def propagate(w, b, X, Y):
    m = X.shape[1]

    # forward prop
    z = np.dot(w.T, X) + b
    a = sigmoid(z)
    cost = (- 1 / m) * np.sum(Y * np.log(a) + (1 - Y) * np.log(1 - a))

    # backward prop
    dz = a - Y
    dw = (1 / m) * np.dot(X, dz.T)
    db = (1 / m) * np.sum(a - Y)

    return cost, dw, db

def gradient_descent(w, b, X, Y, num_iter, alpha, print_cost=False):
    costs = []
    for i in range(num_iter):
        cost, dw, db = propagate(w, b, X, Y)
        w = w - alpha * dw
        b = b - alpha * db
        if i % 100:
            costs.append(cost)
            if print_cost:
                print('Cost after iteration {}: {}'.format(i, cost))
    return costs, w, b

def predict(w, b, X):
    m = X.shape[1]
    Y_pred = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    a = sigmoid(np.dot(w.T, X) + b)
    for i in range(a.shape[1]):
        Y_pred[0][i] = 1 if a[0][i] >= 0.5 else 0
    return Y_pred


def model(X_train, Y_train, X_test, Y_test, num_iter=2000, alpha=0.5, print_cost=False):
    w, b = init_param(X_train.shape[0])
    costs, w , b = gradient_descent(w, b, X_train, Y_train, num_iter, alpha, print_cost)
    Y_pred_train = predict(w, b, X_train)
    print(Y_pred_train)
    Y_pred_test = predict(w, b, X_test)
    print("Train accuracy: {} %".format(100 - np.mean(np.abs(Y_pred_train - Y_train)) * 100))
    print("Test accuracy: {} %".format(100 - np.mean(np.abs(Y_pred_test - Y_test)) * 100))
    values = {"costs": costs,
                "Y_pred_test": Y_pred_test, 
                "Y_pred_train" : Y_pred_train, 
                "w" : w, 
                "b" : b,
                "alpha" : alpha,
                "num_iter": num_iter}
    return values

d = model(X_train, Y_train, X_test, Y_test, num_iter=2000, alpha=0.01, print_cost=True)

costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("alpha =" + str(d["alpha"]))
plt.show()
