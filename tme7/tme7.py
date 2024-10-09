import numpy as np
import random

def learnHMM(allX , allS, N , K):
    #allX : les observations (A , C , G , T)
    #allS : les états (non codante , pos0 , pos1 , pos2)

    #N : nb états => 4
    #K : nb obs => 4
    A = np.zeros((N , N))

    B = np.zeros((N , K))

   

    T = len(allX) 
    for i in range(T-1) :
        A[allS[i]][allS[i+1]] +=  1
        B[allS[i]][allX[i]] += 1
    B[allS[T-1]][allX[T-1]] += 1  
    A /= np.maximum(A.sum(1).reshape(N, 1), 1)  
    B /= np.maximum(B.sum(1).reshape(N, 1), 1)
    return A , B

def viterbi(allx , Pi , A , B) :
    """
    allx : array (T,)
        Sequence d'observations.
    Pi: array, (N,)
        Distribution de probabilite initiale
    A : array (N, N)
        Matrice de transition
    B : array (N, K) // N: états , K : obs
        Matrice d'emission matrix

    """
    N = len(A)
    T = len(allx)
    ## initialisation
    delta = np.zeros((N, T))  
    delta[:,0] = np.log(Pi) + np.log(B[: ,allx[0]])
    psi = np.zeros((N, T)) 
    psi[:,0]= -1


    for t in range(1 , T) :

        for j in range(N) :
            delta[j][t] = np.max([ (delta[i][t-1] + np.log(A[i][j])) for i in range(N)] + np.log(B[j][allx[t]]))
            psi[j][t] = np.argmax([(delta[i][t-1] + np.log(A[i][j]))  for i in range(N)])
    
    etats_predits = np.zeros((T))
    etats_predits[T-1] = int(np.argmax(delta[:,T-1]))
    for t in range(T-2 , -1 , -1) :
        etats_predits[t] = int(psi[int(etats_predits[t+1])][t+1] )
    
    return etats_predits

def get_and_show_coding(etat_predits,annotation_test) :
    return np.where(etat_predits>=1 , 1 , 0) , np.where(annotation_test>=1 , 1 , 0)

def create_confusion_matrix(codants_predits,codants_test):
    r_codantes_index = np.where(codants_test==1)[0]
    r_intergenique_index = np.where(codants_test==0)[0]
    TP = np.sum([1 if codants_predits[i]==1 else 0 for i in r_codantes_index])
    TN = np.sum([1 if codants_predits[i]==0 else 0 for i in r_intergenique_index])
    FN = len(r_intergenique_index) - TN
    FP = len(r_codantes_index) - TP
    return np.array([[TP , FP ],[FN , TN]])

def create_seq(N,Pi,A,B,states,obs) :
    Pi = np.array(Pi)
    A = np.array(A)
    obs = np.array(obs)

    #création d'une matrice de cumule de A
    A_cum = np.zeros((len(A),len(A[0])))
    for i in range(len(A)):
        A_cum[i][0] = A[i][0]
        for j in range(1 , len(A[0])):
            A_cum[i][j] = A_cum[i][j-1] + A[i][j]

    #création d'une matrice de cumule de B
    B_cum = np.zeros((len(B),len(B[0])))
    for i in range(len(B)):
        B_cum[i][0] = B[i][0]
        for j in range(1 , len(B[0])):
            B_cum[i][j] = B_cum[i][j-1] + B[i][j]

    s_1 = states[np.where(Pi==1)[0][0]]
    n = random.uniform(0, 1)
    x = obs[np.where(np.array(B_cum[s_1])>=n)[0][0]]
    for i in range(N) :
        n = random.uniform(0, 1)
        s_2 = states[np.where(np.array(A_cum[s_1])>=n)[0][0]]
        n = random.uniform(0, 1)
        x = obs[np.where(np.array(B_cum[s_2])>=n)[0][0]]
        s_1 = s_2

def get_annoatation2(annotation):
    i = 0
    res = np.zeros((len(annotation)) , dtype=int)
    
    while i<len(annotation) :
        if annotation[i] == 0 :
            if i-1 >= 0 and annotation[i-1]>=1:
                res[i-1] = 9
                res[i-2] = 8
                res[i-3] = 7
            i+=1
        else:
            if (i-1 >=0 and annotation[i-1]==0) or i-1<0 :
                res[i] = 1
                res[i+1] = 2
                res[i+2] = 3
                i+=3
            else :
                res[i] = 4
                res[i+1] = 5
                res[i+2] = 6
                i+=3
    if annotation[i-1]>=1 :
        res[i-1] = 9
        res[i-2] = 8
        res[i-3] = 7

    return res
