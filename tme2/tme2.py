#Anyes TAFOUGHALT 21200397
#Racha Nadine DJEGHALI 21200169

import random
import matplotlib.pyplot as plt
import numpy as np
import math
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb

def bernoulli(p):
  n = random.uniform(0.0, 1.0)
  if n<p :
    return 1
  else:
    return 0

def binomiale(n , p):
  x = 0
  for i in range(n):
    x += bernoulli(p)
  return x

def galton(l, n, p):
  return [binomiale(n , p) for i in range(l)]

def histo_galton(l, n, p):
  res = galton(l, n, p)
  plt.hist(res , len(np.unique(res)))

def densiteNormale(x,m,sigma):
    return 1/(sigma * math.sqrt(2*math.pi))*math.exp(-0.5*((x-m)/sigma)**2)

def normale(k , sigma):
  if k % 2 : 
    x = np.linspace ( -2 * sigma, 2 * sigma, k )
    return [densiteNormale(xi,0,sigma) for xi in x]
  else : 
    raise ValueError("k est pair")

def proba_affine(k, slope):
  if k % 2 :
    x = range(k)
    return [((1/k) + (i - (k-i)/2)*slope) for i in range(k) ]
  else:
    raise ValueError("k est pair")

def Pxy(A , B):
  res = np.zeros((len(A), len(B)), dtype=float) 
  for i in range(len(A)) :
    for j in range(len(B)) :
      res[i][j] = "{:.2f}".format(A[i] * B[j])
  return res

def calcYZ(P_XYZT):
  P_YZ = np.zeros([len(P_XYZT[0]) , len(P_XYZT[0][0])] , dtype=float)
  for y in range(len(P_XYZT[0])):
    for z in range(len(P_XYZT[0][0])):
      for x in range(len(P_XYZT)):
        P_YZ[y][z] += np.sum(P_XYZT[x][y][z]) 
  return P_YZ
  
def calcXTcondYZ(P_XYZT):
  P_YZ = calcYZ(P_XYZT)
  P_XTcondYZ = np.zeros([len(P_XYZT) , len(P_XYZT[0]) , len(P_XYZT[0][0]) , len(P_XYZT[0][0][0])] , dtype=float)
  for x in range(len(P_XYZT)):
    for y in range(len(P_XYZT[0])):
      for z in range(len(P_XYZT[0][0])):
        for t in range(len(P_XYZT[0][0][0])):
          P_XTcondYZ[x][y][z][t] = P_XYZT[x][y][z][t] / P_YZ[y][z]
  return P_XTcondYZ

def calcX_etTcondYZ(P_XYZT):
    P_XcondYZ = np.zeros([len(P_XYZT) , len(P_XYZT[0]) , len(P_XYZT[0][0])] , dtype=float)
    P_TcondYZ = np.zeros([len(P_XYZT[0]) , len(P_XYZT[0][0]) , len(P_XYZT[0][0][0])] , dtype=float)
    P_XTcondYZ = calcXTcondYZ(P_XYZT)

    for x in range(len(P_XYZT)):
      for y in range(len(P_XYZT[0])):
        for z in range(len(P_XYZT[0][0])):
          P_XcondYZ[x][y][z] = np.sum(P_XTcondYZ[x][y][z])
    
    for y in range(len(P_XYZT[0])):
      for z in range(len(P_XYZT[0][0])):
        for t in range(len(P_XYZT[0][0][0])):
          for x in range(len(P_XYZT)):
            P_TcondYZ[y][z][t] +=  P_XTcondYZ[x][y][z][t]
    
    return P_XcondYZ,P_TcondYZ

def testXTindepCondYZ(P_XYZT,epsilon) :
  P_XTcondYZ = calcXTcondYZ(P_XYZT)
  P_XcondYZ,P_TcondYZ = calcX_etTcondYZ(P_XYZT)
  for x in range(len(P_XYZT)):
    for y in range(len(P_XYZT[0])):
      for z in range(len(P_XYZT[0][0])):
        for t in range(len(P_XYZT[0][0][0])):
          if abs(P_XTcondYZ[x][y][z][t] - P_XcondYZ[x][y][z] * P_TcondYZ[y][z][t]) >  epsilon :
            return False
  return True

def testXindepYZ(P_XYZT,epsilon) :
  P_XY = calcYZ(P_XYZT)
  for x in range(len(P_XYZT)):
    for y in range(len(P_XYZT[0])):
      for z in range(len(P_XYZT[0][0])):
        if abs(np.sum(P_XYZT[x][y][z]) - (np.sum(P_XYZT[x]) * P_XY[y][z])) > epsilon :
          return False
  return True

def conditional_indep(P, X, Y, Z, epsilon):
  XYZ = []
  XZ = []
  YZ = []

  XYZ.append(X)
  XYZ.append(Y)
  XZ.append(X)
  YZ.append(Y)

  if len(Z) != 0:
    XYZ.extend(Z)
    XZ.extend(Z)
    YZ.extend(Z)
  if len(Z) != 0:
    PXY_condZ = P.margSumIn(XYZ) / P.margSumIn(Z)
    PX_condZ = P.margSumIn(XZ) / P.margSumIn(Z)
    PY_condZ = P.margSumIn(YZ) / P.margSumIn(Z)
  else:
    PXY_condZ = P.margSumIn(XYZ)
    PX_condZ = P.margSumIn(XZ)
    PY_condZ = P.margSumIn(YZ)

  if (PXY_condZ - (PX_condZ * PY_condZ)).abs().max() > epsilon :
    return False
  return True

def compact_conditional_proba(S , Xn):
  K = S
  K = K.margSumOut(Xn)
  for X in S.var_names :
    if X != Xn:
      K_sans_X = K.margSumOut(X)
      if conditional_indep(S , Xn ,X , K_sans_X.var_names, epsilon=1e-10) :
        K = K_sans_X
  all_var = [Xn]
  all_var.extend(K.var_names)
  p_cond = S.margSumIn(all_var) / K
  return p_cond.putFirst(Xn)

def create_bayesian_network(P , epsilon):
  liste = []
  var_names = P.var_names
  for Xi in reversed(var_names):
    Q = compact_conditional_proba(P , Xi)
    #print(Q.var_names)
    liste.append(Q)
    P = P.margSumOut(Xi)
  return liste

def calcNbParams(P):
  taille_jointe = P.domainSize()
  rb = create_bayesian_network (P, 0.001)
  taille_rb = 0
  for r in rb:
    taille_rb += r.domainSize()
  return taille_jointe, taille_rb



  






