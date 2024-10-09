import numpy as np
import utils
import scipy.stats as stats
import math

def sufficient_statistics(data , dico , x , y , Z):
    resultat=utils.create_contingency_table( data, dico, x, y, Z )
    A = 0.0
    for i in range(len(dico[x])) :
        for j in range(len(dico[y])) :
            for k in range(len(resultat)):
                if resultat[k][0] != 0 :
                    v = np.sum(resultat[k][1][i]) * np.sum(resultat[k][1][:,j]) / resultat[k][0]
                    if v != 0 :
                        A += ( resultat[k][1][i][j] - v ) ** 2 / v  
    degree = (len(dico[x])-1) * (len(dico[y])-1) * np.sum([1 if z[0]!=0 else 0 for z in resultat])      
    return (A , degree)

def indep_score(data , dico , x , y , Z):
    if len(data) < len(dico[x]) * len(dico[y]) * len(Z):
        return -1
    else:
        A , degree = sufficient_statistics(data , dico , x , y , Z)
        return stats.chi2.sf (  A, degree )
    
def best_candidate(data , dico , x , Z , alpha):
    mini = math.inf
    mini_i = []
    for i in range(x):
        if i not in Z :
            p_value = indep_score(data , dico , x , i , Z)
            if p_value < alpha and p_value < mini :
                mini = p_value
                mini_i = [i]
    return mini_i

def create_parents(data , dico , x , alpha) :
    Z = []
    y = best_candidate(data , dico , x , Z , alpha)
    while(len(y)!=0):
        Z.extend(y)
        y = best_candidate(data , dico , x , Z , alpha)
    return Z

def learn_BN_structure ( data , dico , alpha ) :
    parents = []
    for x in range(len(dico)):
        parents.append(create_parents(data , dico , x , alpha))
    return parents
    