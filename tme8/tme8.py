import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0) 

def gen_data_lin(a, b, sig, N,Ntest):

    X_train = np.sort(np.random.rand(N))
    X_test = np.sort(np.random.rand(Ntest))

    Y_train = a*X_train + b + np.random.normal(0,sig ,N) 
    Y_test = a*X_test + b + np.random.normal(0,sig , Ntest) 

    return X_train, Y_train, X_test, Y_test 
    

def modele_lin_analytique(X_train, y_train):
    cov = np.cov(X_train , y_train ,bias=True)
    esp_X = np.mean(X_train)
    esp_Y = np.mean(y_train)

    return cov[0][1]/cov[0][0] , esp_Y - cov[0][1]*esp_X/cov[0][0]

def calcul_prediction_lin(X,ahat,bhat):
    return ahat * X + bhat

def erreur_mc(y_train, yhat_train) :
    return np.mean((y_train - yhat_train)**2)

def dessine_reg_lin(X_train, y_train, X_test, y_test,a,b):
    plt.plot(X_test, y_test, 'r.',alpha=0.2,label="test")
    plt.plot(X_train, y_train, 'b',label="train")
    plt.plot(X_test, calcul_prediction_lin(X_test,a,b), 'g',label="prediction")
    plt.legend()

def make_mat_lin_biais(X):
    return np.hstack((X.reshape(len(X),1), np.ones((len(X), 1))))

def reglin_matriciel(Xe,y):
    return np.linalg.solve(Xe.T @ Xe , Xe.T @ y)

def calcul_prediction_matriciel(Xe,w):
    return Xe @ w


def gen_data_poly2(a, b, c, sig, N=100, Ntest=500) :
    X_train = np.sort(np.random.rand(N))
    X_test = np.sort(np.random.rand(Ntest))

    Y_train = a*(X_train**2) + b*X_train + c + np.random.normal(0,sig ,N) 
    Y_test = a*(X_test**2) + b*X_test + c  + np.random.normal(0,sig , Ntest) 

    return X_train, Y_train, X_test, Y_test 

def make_mat_poly_biais(Xp):
    N = len(Xp)
    Xpe = np.ones((N , 3))
    Xpe[:,0] = Xp**2
    Xpe[:,1] = Xp
    return Xpe

def dessine_poly_matriciel(Xp_train,yp_train,Xp_test,yp_test,w) :
    plt.plot(Xp_test, yp_test, 'r.',alpha=0.2,label="test")
    plt.plot(Xp_train, yp_train, 'b',label="train")
    Xpe_test = make_mat_poly_biais(Xp_test)
    plt.plot(Xp_test, calcul_prediction_matriciel(Xpe_test,w), 'g',label="prediction")
    plt.legend()


def descente_grad_mc(Xe, y, eps=1e-4, nIterations=500) :
    allW = []
    w_1 =np.zeros(len(Xe[0]))
    for i in range(nIterations) :
        allW.append(w_1)
        w_2 = w_1.copy()
        w_2 -= eps * 2 * (Xe.T @ (Xe @ w_1 - y))
        if np.array_equal(w_1,w_2) :
            w_1 , np.array(allW)
        w_1 = w_2.copy()

    allW.append(w_1)
    return w_1 , np.array(allW)

def application_reelle(X_train,y_train,X_test,y_test):
    w = reglin_matriciel(X_train,y_train)
    yhat= calcul_prediction_matriciel(X_train,w)
    yhat_t= calcul_prediction_matriciel(X_test,w)
    print(f'Erreur moyenne au sens des moindres carrés (train): {erreur_mc(yhat, y_train)=:.4}')
    print(f'Erreur moyenne au sens des moindres carrés (test): {erreur_mc(yhat_t, y_test)=:.4}')
    return w, yhat , yhat_t

def normalisation(X_train, X_test) :
    mu = X_train.mean()
    std = X_train.std()
    Xn_train = (X_train - mu ) / std
    Xn_test = (X_test - mu ) / std
    return np.column_stack((Xn_train,np.ones((len(X_train))))),np.column_stack((Xn_test,np.ones((len(X_test)))))

