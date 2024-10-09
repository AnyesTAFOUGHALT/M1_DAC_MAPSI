import numpy as np
import os
import pickle as pkl



def normale_bidim( x , mu , Sig ):
    coef = 1/( (2*np.pi)**(len(x)/2) * np.linalg.det(Sig)**0.5 )
    return coef * np.exp(-0.5 * np.matmul(np.matmul((x - mu) , np.linalg.inv(Sig)) , np.transpose(x - mu)))

#def estimation_nuage_haut_gauche():

def init(X) :
    pi = np.array([0.5 , 0.5])
    mean_X0 = np.mean(X[:,0])
    mean_X1 = np.mean(X[:,1])
    mu = np.array([[mean_X0+1 , mean_X1+1] , [mean_X0-1 , mean_X1-1] ])
    sig = np.array([np.cov(X.T), np.cov(X.T)])
    return pi , mu , sig

def Q_i(X, pi, mu, Sig) :
    Q = []
    for i in range(X.shape[0]) :
        Y = []
        for k in range(len(pi)):
            Y.append(normale_bidim( X[i] , mu[k] , Sig[k]) * pi[k]/ np.sum([normale_bidim( X[i] , mu[j] , Sig[j]) * pi[j] for j in range(len(pi))]))    
        Q.append(Y)
    return np.transpose(Q)

def update_param(X, q, pi, mu, Sig) :
    pi_u = np.zeros([len(pi)] , dtype = float)
    mu_u = np.zeros([mu.shape[0] , mu.shape[1]] , dtype = float)
    Sig_u = np.zeros([len(pi), Sig.shape[1] , Sig.shape[2]] , dtype = float)
    q = Q_i(X, pi, mu, Sig)
    for i in range(len(pi)):
            q_i = np.sum(q[i])
            pi_u[i] = q_i / np.sum(q)
            mu_u[i] = np.sum( [np.dot(q[i][j] , X[j]) for j in range(len(X))],axis=0) / q_i
            Sig_u[i] = np.matmul((q[i] * (X-mu_u[i]).T) , (X-mu_u[i])) / q_i
    return pi_u , mu_u , Sig_u

def EM(X, initFunc=init, nIterMax=100, saveParam=None , epsilon = 1e-3):
    pi, mu, Sig = initFunc(X)
    pi_new, mu_new, Sig_new = [] , [] , []
    for i in range(nIterMax) :
        q = Q_i(X, pi, mu, Sig)
        pi_new, mu_new, Sig_new = update_param(X, q, pi, mu, Sig)
        if saveParam is not None:                                         # détection de la sauvergarde
            if not os.path.exists(saveParam[:saveParam.rfind('/')]):     # création du sous-répertoire
                 os.makedirs(saveParam[:saveParam.rfind('/')])
            pkl.dump({'pi':pi_new, 'mu':mu_new, 'Sig': Sig_new}, open(saveParam+str(i)+".pkl",'wb'))                 # sérialisation
 
        if np.abs(mu_new - mu).sum() < epsilon:
            return i , pi_new, mu_new, Sig_new 
        pi, mu, Sig = pi_new, mu_new, Sig_new
    return nIterMax , pi_new, mu_new, Sig_new 

def init_4(X) :
    pi = np.array([0.25 , 0.25 , 0.25 , 0.25])
    mean_X0 = np.mean(X[:,0])
    mean_X1 = np.mean(X[:,1])
    mu = np.array([[mean_X0 + 1,mean_X1+1],[mean_X0+1,mean_X1-1], [mean_X0-1,mean_X1+1],  [mean_X0-1,mean_X1-1] ])
    sig = np.array([np.cov(X.T), np.cov(X.T) , np.cov(X.T), np.cov(X.T)])
    return pi , mu , sig

def bad_init_4(X):
    pi = np.array([0.25 , 0.25 , 0.25 , 0.25])
    mean_X0 = np.mean(X[:,0])
    mean_X1 = np.mean(X[:,1])
    mu = np.array([[mean_X0 + 4,mean_X1+2],[mean_X0+3,mean_X1+4], [mean_X0,mean_X1],  [mean_X0-5,mean_X1] ])
    sig = np.array([np.cov(X.T), np.cov(X.T) , np.cov(X.T), np.cov(X.T)])
    return pi , mu , sig