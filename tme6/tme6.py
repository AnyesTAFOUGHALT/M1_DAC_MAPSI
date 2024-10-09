import numpy as np

def discretise(X, d):
    intervalle = 360 / d
    return [np.floor(y/intervalle) for y in X]

def groupByLabel(Y) :
    classes = np.unique(Y)
    return {lettre: np.array(np.where(Y==lettre)[0]) for lettre in classes}

def learnMarkovModel(X , d) :
    A = np.zeros((d , d))
    Pi = np.zeros((d))

    #on déscritise
    Xd = discretise(X, d)
    for signal in Xd :
        Pi[int(signal[0])] += 1
        for i in range(len(signal)-1) :
            A[int(signal[i])][int(signal[i+1])] +=  1
    A /= np.maximum(A.sum(1).reshape(d, 1), 1)
    Pi = Pi / Pi.sum()

    return Pi , A

def learn_all_MarkovModels(X,Y,d) :
    classes = np.unique(Y)
    return {lettre: learnMarkovModel([X[i] for i in groupByLabel(Y)[lettre]] , d) for lettre in classes}

def stationary_distribution_freq(Xd , d):
    X = np.concatenate(Xd, axis=0)
    return np.array([len(np.where(X == float(i))[0])  for i in range(d) ]) / len(X)

def stationary_distribution_sampling(Pi , A , N , eps = 0.001 ):
    P = Pi
    for i in range(N) :
        print(P)
        Q = np.dot(P , A)
        if np.sum(np.abs(P - Q) <= eps ) == 0:
            return Q
        P = Q
    return Q

def stationary_distribution_fixed_point(A , epsilon) :
    B = np.dot(A , A)
    C = A
    while np.square(np.subtract(C, B)).mean() > epsilon :
        C = B
        B = np.dot( B, A)
    return np.array([B[0][i] for i in range(len(B[0]))])

def stationary_distribution_fixed_point_VP(A, epsilon=1e-8):
    A_transpose = A.T
    eigenvalues, eigenvectors = np.linalg.eig(A_transpose)
    
    idx = np.argmin(np.abs(eigenvalues - 1))
    stationary_distribution = np.real(eigenvectors[:, idx])
    stationary_distribution /= np.sum(stationary_distribution)
    
    return stationary_distribution.reshape(len(stationary_distribution) , 1)

def logL_Sequence(s , Pi , A) :
    logL = np.log(Pi[int(s[0])])
    return logL + np.sum( [ np.log(A[int(s[i])][int(s[i+1])]) for i in range(len(s) -1)] )

def compute_all_ll(Xd,models):
    return [[logL_Sequence(s , *models[k]) for s in Xd ] for k in models]

def accuracy(ll,Y) :
    l = np.array(ll)
    nb_true = np.sum([1 if (np.argmax(l[ : , i]) == np.where(np.unique(Y) == Y[i]) ) else 0 for i in range(len(Y))])
    return nb_true / len(Y)

def learnMarkovModel_Laplace(X , d) :
    A = np.ones((d , d))
    Pi = np.ones((d))

    #on déscritise
    Xd = discretise(X, d)
    for signal in Xd :
        Pi[int(signal[0])] += 1
        for i in range(len(signal)-1) :
            A[int(signal[i])][int(signal[i+1])] +=  1
    A /= np.maximum(A.sum(1).reshape(d, 1), 1)
    Pi = Pi / Pi.sum()

    return Pi , A

def learn_all_MarkovModels_Laplace(X,Y,d) :
    classes = np.unique(Y)
    return {lettre: learnMarkovModel_Laplace([X[i] for i in groupByLabel(Y)[lettre]] , d) for lettre in classes}
