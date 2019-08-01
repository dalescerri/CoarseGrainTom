import itertools
import numpy as np
import scipy.linalg as lin
import copy
import math as mt
import cmath as cp
import operator
import time
import timeit
import datetime
import six
import pickle
import itertools
from qutip import *
from scipy.optimize import least_squares
from libs_rand import Funs as funs
from libs_rand import DataGen as datagen
from importlib import reload
reload (funs)
reload (datagen)
from joblib import Parallel, delayed
import multiprocessing

num_cores = -1

def save_dict(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dictionary, f)

def load_dict(filename):
    with open(filename, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

Nq = 1
tom = funs.TomFuns()
tom.set_tomography(Nqubits=Nq)


i = complex(0,1)
count = 0
aver = 1000
sorting=False
dist='uniform'

NoClickArr = [100,1000,10000]

if dist=='exponential':
    Beta = [0,np.pi/16,np.pi,10000*np.pi]
else:
    Beta = [0]
    
if sorting:
    Segments = [4,8,16]
else:
    Segments = [0]

rho = [rand_dm_ginibre(2**Nq,rank=None).full() for j in range(aver)]

flagcounter = []
flag = 100

def run_exp(N,Inc,beta,Av):

    costfunction = []
    gendata = datagen.DataGen()
    gendata.set_experiment(NoClicks=N,Slices=Inc,Nqubits=Nq,Rho=rho[Av],dist_key=dist,scale_par=beta,sort=sorting)

    Projs, Nvr2 = gendata.Mvr(), gendata.Nvr()
    Mvr = Projs[1]
    State = Projs[0]

    Nvr = np.multiply(Nvr2,N**-1)

    done = 0


    r = N
    gamma = 1e-3
    kount = 0
    rhoEst = (.5)*np.eye(2**Nq)
    tol = .1e-4
    maxit=2000


    it = 0
    itert = 0
    fe=0
    start = timeit.default_timer()
    while done == 0:

        it+=1;
        if (itert>=1):
            g0 = g

        f,g = tom.costfun(Nq,State,Nvr,rhoEst)

        costfunction.append(f)
        normg = lin.norm(g,'fro')

        if itert==0:
            fm = f

        fe+=1

        mu=1.0

        if itert>=1:
            mu = min( max(np.trace(Dj.dot(g-g0))/(t*normD**2),1e-4), 1e4)

        Ai = rhoEst - (1/mu)*g;

        Di,Vi = np.linalg.eig(Ai)
        x = tom.SimplexProjVec(Di)

        Di = np.diag(Di)

        Pa = Vi.dot(np.diag(x).dot(Vi.conj().T))
        Pa = (Pa+ Pa.conj().T)/2

        Dj = Pa - rhoEst
        normD = lin.norm(Dj,'fro')

        gtd = np.trace(g.dot(Dj)).real

        if gtd > -tol*1e-11:
            if (normD < 1e-8):
                done=1
                flag=2
                continue
            else:
                done=1
                flag=-3

        if normD < tol:
            done=1
            flag=1
            #continue

        t=1

        rhoEstn = rhoEst + t*Dj
        fn = tom.costfun(Nq,State,Nvr,rhoEstn)[0]

        while round(fn - (fm + gamma*t*gtd),3) > 0:
            t=0.5*t

            if t<1e-8:
                done=1
                flag=-2
                break


            if it>100 and costfunction[-1]<2 and np.mean(abs(np.diff(costfunction[-20:-1])))<1e-4:
                done=1
                print('new exit')

            rhoEstn = rhoEst + t*Dj
            fn = tom.costfun(Nq,State,Nvr,rhoEstn)[0]
            fe+=1

        rhoEst=rhoEstn
        f = fn

        itert+=1

        fm = fn

        if itert>maxit:
            done=1
            flag=-1

    stop = timeit.default_timer()

    mat1_Jac = lin.sqrtm(rho[Av])
    mat2_Jac = np.dot(mat1_Jac,rhoEst)
    mat3_Jac = lin.sqrtm(np.dot(mat2_Jac,mat1_Jac))

    return [np.trace(mat3_Jac).real, stop - start]

Test={}

for N in NoClickArr:
    for Inc in Segments:
        for b in Beta:
            Test[(N,Inc,b)] = []

for N in NoClickArr:
    for Inc in Segments:
        for b in Beta:
            Test[(N,Inc,b)].append(Parallel(n_jobs=num_cores)(delayed(run_exp)(N,Inc,b,k) for k in range(aver)))
            save_dict(Test,'PGDB_uniform')
