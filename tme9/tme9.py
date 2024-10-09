import numpy as np
import matplotlib.pyplot as plt

def labels_tobinary(Y , cl):
    new_Y = np.where(Y==cl , 1. , 0.)
    return new_Y

def pred_lr(X, w,b):
    return [1 / ( 1 + np.exp(-(w@X[i].T + b))) for i in range(len(X))]

def classify_binary(Y_pred):
    cl =  np.where(np.array(Y_pred)>0.5 , 1. , 0.)
    return cl

def accuracy(Y_predb, Y_c):
    return np.sum(np.where(np.array(Y_predb)==np.array(Y_c) , 1 ,0))/len(Y_predb)

def rl_gradient_ascent(X,Y,eta = 1e-4, niter_max=300) :
    N , d = X.shape
    w_1 = np.zeros(d)
    b_1 = 0
    accs = []
    for i in range(niter_max) :
        pred = 1 / ( 1 + np.exp(-( w_1@X.T + b_1)))
        accs.append(accuracy(classify_binary(pred_lr(X, w_1,b_1)) , Y))
        w_2 = w_1 + eta * (X.T @ (Y.T - pred))
        b_2 = b_1 + eta * np.sum(Y - pred)
        if np.array_equal(w_1, w_2) and b_1==b_2 :
            return w_1 , b_1 , accs , i
        w_1 = w_2.copy()
        b_1 = b_2
    return w_1 , b_1 , accs , niter_max

def visualization(w) :
    plt.figure()
    plt.imshow(w.reshape(16,16), cmap='gray')

def rl_gradient_ascent_one_against_all(X,Y,epsilon = 1e-4, niter_max=1000):
    allW = []
    allB = []
    Yunique = np.unique(Y)

    for cl in Yunique :
        new_Y = labels_tobinary(Y , cl)
        w , b , accs , iter = rl_gradient_ascent(X,new_Y,epsilon, niter_max)
        allW.append(w)
        allB.append(b)
        print("Classe : ",cl," acc train={0:.2f} %".format(accs[-1]*100))
    return np.array(allW) , np.array(allB)

def classif_multi_class(Y):
    return np.argmax(Y , axis=1)


def normalize(X):
    return X - 1


def pred_lr_multi_class(X, w, b):
    s = X @ w + b
    return [[np.exp(s[i][k])/(np.sum([np.exp(s[i][j]) for j in range(len(b))])) for k in range(len(b)) ]for i in range(len(X))]

def to_categorical(Y, K):
    return np.eye(K)[Y]