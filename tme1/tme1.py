import numpy as np
import matplotlib.pyplot as plt

def analyse_rapide(x):
  print("mean:", np.mean(x))
  print("std:", np.std(x))
  print("quantile:", np.quantile(x , np.arange(0 , 1 , 0.1)))

def discretisation_histogramme(d,n):
  sup = max(d)
  inf = min(d)
  pas = (sup-inf)/n
  bornes = np.arange(inf , sup , pas)
  print( "Bornes :" , bornes)
  effectifs = [np.where((d>bornes[i]) & (d<bornes[i+1]), 1, 0).sum() for i in range(len(bornes)-1) ]
  print("Effectifs:",effectifs)
  #plt.axis([0,2000, 0, 1600])
  x = [1, 2, 2, 3, 4, 4
, 4, 4, 4, 5, 5]
  #plt.bar(bornes, effectifs,width=0.2)
  #plt.show()
