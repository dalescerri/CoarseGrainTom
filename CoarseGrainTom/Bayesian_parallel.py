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

num_cores = 1

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
aver = 1000
sorting = False
dist='uniform'

NoClickArr = [100,1000,10000]
if dist=='exponential':
    Beta = [0,np.pi/16,np.pi/8,np.pi/4,np.pi/2,10000*np.pi]
else:
    Beta = [0]
    
if sorting:
    Segments = [4,8,16]
else:
    Segments = [0]

rho = [rand_dm_ginibre(2**Nq, None).full() for j in range(aver)]

def run_exp(N,beta,Inc,Av):
    
    gendata = datagen.DataGen()
    gendata.set_experiment(NoClicks=N,Slices=Inc,Nqubits=Nq,Rho=rho[Av],resamp=False,n_samples=1000,
                           dist_key=dist,scale_par=beta,sort=sorting)

    Projs, Nvr2 = gendata.Mvr(), gendata.Nvr()
    Mvr = Projs[1]
    State = Projs[0]

    start = timeit.default_timer()
    rhoEst = gendata.update_weights()
    stop = timeit.default_timer()

    mat1_Jac = lin.sqrtm(rho[Av])
    mat2_Jac = np.dot(mat1_Jac,rhoEst)
    mat3_Jac = lin.sqrtm(np.dot(mat2_Jac,mat1_Jac))
    print('Fidelity',np.trace(mat3_Jac).real)
    
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
            save_dict(Test,'Bayesian_uniform')