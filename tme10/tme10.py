#DJEGHALI RACHA NADINE 
#TAFOUGHALT ANYES
import numpy as np

def exp (rate):
    rate = np.where(rate == 0, 1e-200, rate)
    #x=np.random.rand(*rate.shape)
    u = np.random.rand(*rate.shape)
    return - np.log(1-u) / rate


    
    
def simulation(graph , sources , maxT) :
    nodes, k, r = graph
    N = len(nodes)
    
    temps_infection = np.array([ float(maxT) if i not in sources else 0 for i in range(N)])
   
    infectieux = temps_infection.copy()
    
    while True:
        i = infectieux.argmin()
        if infectieux[i] >= maxT :
            break 

        infectieux[i] = maxT 

        t_i = temps_infection[i]

        cibles_index  = np.where(temps_infection > t_i)[0]
        cibles_k=np.array([k.get((i, j),0) for j in cibles_index])
        cibles_r=np.array([r.get((i, j),0) for j in cibles_index] )
        
        x = np.random.random(len(cibles_k)) < cibles_k
        delta = exp(cibles_r * x )
        t = t_i + delta
        mini = np.minimum(t , temps_infection[cibles_index])
        temps_infection[cibles_index] = mini
        infectieux[cibles_index] = mini
    
    return temps_infection

def getProbaMC(graph,sources,maxT,nbsimu):
    names = graph[0]
    nombreNoeuds = len(names)
    probas = np.zeros(nombreNoeuds)
    for i in range(nbsimu):
        ti = simulation(graph,sources,maxT)
        probas = probas + np.where((0 < ti) & (ti < maxT), 1, 0)
    return probas/nbsimu

def getPredsSuccs(graph):
    preds = {}
    succs = {}
    
    k , r = graph[1], graph[2]
    arcs =list(k.keys())
    for i in range(len(arcs)):
        arc = arcs[i]
        prec= preds.get(arc[1],[])
        prec.append((arc[0],k[arc],r[arc]))
        preds[arc[1] ]= prec
        suiv= succs.get(arc[0],[])
        suiv.append((arc[1],k[arc],r[arc]))
        succs[arc[0]]= suiv

    return preds,succs

def compute_ab(v,times,preeds,maxT,eps=1e-20):
    if times[v]==0:
        return 1,0
        
    a = 1
    preds = preeds[v]
    biv = [kiv  * np.exp(-(riv *(times[v]-times[i])))+ 1 - kiv for  i,kiv,riv in preds  
           if  times[i] < times[v]  ]

    if times[v]<maxT :
        a = eps
        aiv = [ kiv *riv  * np.exp(-(riv*(times[v]-times[i])))  for i,kiv,riv in preds  
               if times[i] < times[v] ]
        new = np.sum(np.array(aiv)/np.array(biv))
        if new > eps:
            a = new

    return  a,np.sum(np.log(np.array(biv)))

def compute_ll(times,preds,maxT):
    sa = []
    sb = []
    ll =0
    for i in range(len(times)):
        a,b = compute_ab(i,times,preds,maxT)
        sa.append(a)
        sb.append(b)
        ll+= np.log(a)+b

    return ll,sa,sb

def addVatT(v,times,newt,preds,succs,sa,sb,maxT,eps=1e-20):
    succs=succs.get(v,[])
    t=times[v]
    times[v]=newt
    t = newt
    if len(succs)>0:
        c,k,r=map(np.array,zip(*succs))
        tp=times[c]
        which=(tp>t)

        tp=tp[which]
        dt=tp-t
        k=k[which]
        r=r[which]
        c=c[which]
        rt = -r*dt
        b1=k*np.exp(rt)
        b=b1+1.0-k
        
        a=r*b1
        a=a/b
        b=np.log(b)
        
        sa[c]=sa[c]+np.where(tp<maxT,a,0.0)
        sa[c]=np.where(sa[c]>eps,sa[c],eps)
        sb[c]=sb[c]+b
        sb[c]=np.where(sb[c]>0,0,sb[c])
    sa[v],sb[v]=compute_ab(v,times,preds,maxT)


def logsumexp(X):
    x_max = np.max(X)
    return x_max+ np.log(np.sum([np.exp(x-x_max) for x in X ],-1))

def compute_mse(ref, times):
    return np.mean((ref - times) ** 2)


def gb(graph, infections, maxT, sampler, burnin=100, ref=None, period=1000):
    preds, succs = getPredsSuccs(graph)
    nbNodes = len(graph[0])
    times = np.array([maxT] * nbNodes, dtype=float)
    l = list(range(nbNodes))

    for infecte in infections:
        times[infecte[0]] = infecte[1]
        l.remove(infecte[0])

    _, sa, sb = compute_ll(times, preds, maxT)

    rate_list = []

    for i in range(burnin + period):
        v = np.random.choice(l)
        new_time = sampler(v, times, preds, succs, sa, sb, 10, 10, maxT)  
        times[v] = new_time
        ll, sa, sb = compute_ll(times, preds, maxT)

        if ref is not None and i % period == 0:
            mse = compute_mse(ref, times)
            print(f'{i} {times} MSE = {mse}  ll = {ll}')

        if i >= burnin:
            rate_list.append(times.copy())

    rate = np.mean(np.array(rate_list), axis=0)
    print("------")
    print(f"rate={rate}")

    return rate
