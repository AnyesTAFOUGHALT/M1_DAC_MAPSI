import numpy as np
import math
import matplotlib.pyplot as plt

NB_CLASSES = 10
NB_PIXELS = 256

def densiteNormale(x,m,sigma):
    return 1/(sigma * math.sqrt(2*math.pi))*math.exp(-0.5*((x-m)/sigma)**2)

def learnML_parameters( X_train , Y_train ):
  mu = np.zeros([NB_CLASSES , NB_PIXELS] , dtype=float)
  sig = np.zeros([NB_CLASSES , NB_PIXELS] , dtype=float)
  for i in range(NB_CLASSES):
    for j in range(NB_PIXELS):
      mu[i][j] = "{:.8f}".format(np.mean(X_train[Y_train == i][:,j]))
      sig[i][j] = "{:.8f}".format(np.std(X_train[Y_train == i][:,j]))
  return mu , sig
      
def log_likelihood(X , mu , sig , defeps) :
  if defeps > 0 :
    sig = np.array([np.maximum(sig[i] , defeps) for i in range(len(sig))])
  log_vraisemb = 0
  for i in range(NB_PIXELS):
    log_vraisemb += 0 if(defeps == -1 and sig[i] == 0) else np.log(2 * np.pi * sig[i]**2) + ((X[i] - mu[i]) / sig[i])**2
  return - 0.5 * log_vraisemb

def classify_image(X , mu, sig, defeps) :
  vraisemblance = np.array([log_likelihood(X , mu[i] , sig[i] , defeps) for i in range(NB_CLASSES)])
  return np.argmax(vraisemblance)

def classify_all_images(X , mu , sig , defeps) :
  return np.array([ classify_image(X[i] , mu, sig, defeps) for i in range(len(X))] )

def matrice_confusion(Y, Y_hat) :
  mat_conf = np.zeros([NB_CLASSES , NB_CLASSES])
  for i in range(len(Y)) :
      mat_conf[Y[i]][Y_hat[i]] += 1
  return mat_conf

def classificationRate(Y, Y_hat) :
  rate = 0
  for i in range(len(Y)) :
    rate += 1 if Y[i]==Y_hat[i] else 0
  return rate / len(Y)

def classifTest(X_test,Y_test,mu,sig, defeps) :
  print("1- Classify all test images ...")
  Y_test_hat = classify_all_images(X_test , mu , sig , defeps)

  print("Classification rate : ", classificationRate(Y_test, Y_test_hat))
  print("3- Matrice de confusion : ")
  plt.figure(figsize=(3,3))
  plt.imshow(matrice_confusion(Y_test, Y_test_hat));

  return np.where(Y_test!=Y_test_hat)

def binarisation(X):
  Xb = np.zeros([len(X) , NB_PIXELS])
  for i in range(len(X)):
    for j in range(NB_PIXELS):
      Xb[i][j] = 1 if X[i][j] > 0 else 0
  return Xb

def learnBernoulli(Xb , Y) :
  theta = np.zeros([NB_CLASSES , NB_PIXELS] , dtype = float)
  for i in range(NB_CLASSES):
    for j in range(NB_PIXELS):
      X = Xb[np.where(Y==i)]
      theta[i][j] = np.sum(X[:,j]) / np.sum(len(X))
  return theta

def logpobsBernoulli(X , theta , epsilon):
    logprob = np.zeros([NB_CLASSES] , dtype=float)
    for i in range(NB_CLASSES) :
        for j in range(NB_PIXELS):
            if theta[i][j] == 0 :
                p = epsilon
            elif theta[i][j] == 1:
                p = 1 - epsilon
            else :
                p = theta[i][j]
            logprob[i] += (X[j] * np.log(p)+ (1 - X[j]) * np.log(1 - p ))
    return logprob
"""
    Réponse à la question suivante : Ce résultat vous parait-il normal? Qu'est ce qui peut expliquer cette valeur étonnante?
    Oui ce résultat est totalement cohérent car la vraissemblance maximale correspond bien à la classe 0 qui est la prédiction
    de la premiére image
"""

def classify_all_images_Bernoulli(X , theta , epsilon) :
  return np.array([ np.argmax(logpobsBernoulli(X[i] , theta , epsilon)) for i in range(len(X))] )

def classifBernoulliTest(Xb_test , Y_test , theta , epsilon = 1e-4) :
  print("1- Classify all test images ...")
  Y_test_hat = classify_all_images_Bernoulli(Xb_test , theta , epsilon )

  print("Classification rate : ", classificationRate(Y_test, Y_test_hat))
  print("3- Matrice de confusion : ")
  plt.figure(figsize=(3,3))
  plt.imshow(matrice_confusion(Y_test, Y_test_hat));

  return np.where(Y_test!=Y_test_hat)

def learnGeom(Xb , Y , seuil) :
  theta = np.zeros([NB_CLASSES , 16] , dtype = float)
  for i in range(NB_CLASSES):
    for j in range(16):
      X = Xb[np.where(Y==i)]
      theta[i][j] = np.sum(len(X)) / np.sum(X[:,j]) 
  return theta

def logpobsGeom(X , theta ):
    return [np.sum(np.log(theta[i])+ (X - 1) * np.log(1 - theta[i] ))  for i in range(NB_CLASSES) ]

def classifyGeom(X , theta ) :
  return  np.argmax(logpobsGeom(X , theta ))

