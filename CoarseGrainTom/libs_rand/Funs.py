
# coding: utf-8

# In[59]:

import itertools
import numpy as np
import scipy.linalg as lin
import math as mt
import cmath as cp
import operator
import pickle
import itertools
import time
#from numpy import random as rand
from qutip import *
from scipy import *
from copy import deepcopy


class GenFuns():
    
    def __init__(self):
        self.H=np.array([1,0])
        self.V=np.array([0,1])
        self.T = 1/np.sqrt(2)*np.add(self.H,self.V)
        self.B = 1/np.sqrt(2)*np.subtract(self.H,self.V)

        self.sb = [[self.H,self.V],[self.T,self.B]] #single basis
        
        
    def set_tomography(self, Nqubits):
        '''
        Set tomography details.
        
        Imput:
        Nqubits  [int]  :  Number of qubits
        
        '''
        
        self.Nqubits = Nqubits
        
    
    def save_dict(self,dictionary, filename):
        with open(filename, 'wb') as f:
            pickle.dump(dictionary, f)

    def load_dict(self,filename):
        with open(filename, 'rb') as f:
            ret_di = pickle.load(f)
        return ret_di

    
    def tupleless(self,a,b):
        '''
        Elemntwise-comparison between two tuples.
        
        Input:
        a, b [tuple] : Tuples to be compared
        
        Output:
        Boole  :  True if a[j] < b[j] for all j
                  False if not
                  
        '''
        
        count = 0
        for j in range(len(a)):
            if a[j] < b[j]: count+=1

        if count==len(a):
            return True
        else:
            return False

        
    def Centroid(self,b,inc):
        '''
        Calculate centroid.

        Input:
        b         [list / array]  : coordinates
        inc       [int]           : number of segments for each single qubit basis setting

        Output:
        CentroidArr     [array]  : centroid coordinates

        '''

        a = deepcopy(b)
        self.CentroidArr = np.zeros((2,inc))
        for j in range(inc):  
            if len(a[j][0])==0:
                self.CentroidArr[0][j] = 0
                self.CentroidArr[1][j] = 0

            elif len(a[j][0])==1:    #NEVER THE CASE SINCE IF NOT EMPTY, POLYCOORDS HAS AT LEAST 2PTS LEFT FOR COMPLETENESS
                self.CentroidArr[0][j] = a[j][0][0]
                self.CentroidArr[1][j] = a[j][1][0]

            #SAME ANSWER AS ABOVE FOR POLYCOORDS the 3 case means it's still a line due to repeated values
            else:   
                self.CentroidArr[0][j] = (1/len(a[j][0]))*sum(a[j][0][k] for k in range(len(a[j][0])))
                self.CentroidArr[1][j] = (1/len(a[j][1]))*sum(a[j][1][k] for k in range(len(a[j][1])))
        return self.CentroidArr

    
    def Proj(self,x):
        '''
        Returns Trace 1 projection operator of input array x.
        
        '''
        
        if np.trace(np.outer(x,np.conj(x)))==0:
            self.y=np.outer(x,np.conj(x))
        else:
            self.y=np.outer(x,np.conj(x))
        return self.y


    def MultiBasis(self):
        '''
        Generates multi-qubit basis states
        
        Output:
        mb [array]  :  Multi-qubit basis states
        mp [array]  :  Projection operators corresponding to mb
        
        '''

        self.mbl = []
        for basisset in list(itertools.product([0,1], repeat=self.Nqubits)):
            self.mbl += list(itertools.product(*[self.sb[basisset[j]] for j in range(self.Nqubits)])) #list of multiple basis elements

        self.mb = {}
        self.mp = {}
        for basiselem in range(len(self.mbl)):
            self.mb[basiselem] = tensor([Qobj(singlevec) for singlevec in self.mbl[basiselem]]).full()
            self.mb[basiselem].shape = (len(self.mb[basiselem]))
            self.mp[basiselem] = np.outer(self.mb[basiselem],np.conj(self.mb[basiselem]))

        return self.mb, self.mp
    
    
    def SimplexProj(self,rho_in):
        '''
        Projection of rho_in onto the unit simplex
        
        Input:
        rho_in [array] : density matrix to be projected
        
        Output:
        rho_out [array] : projected density matrix
        
        '''

        In = [j for j in range(len(rho_in))]
        x = sorted(lin.eigvals(rho_in).real,reverse=True)
        U = np.array([xel for _, xel in sorted(zip(lin.eigvals(rho_in).real, lin.eig(rho_in)[1]), key=lambda pair: pair[0] ,reverse=True)])

        I = []
        nI = len(x) - len(I)
        xt = [x[j] + (1-sum(x[k] for k in In if k not in I))/nI for j in range(len(x))]
        while len([np.real(el) for el in xt if el<0]) >0:
            for j in range(len(x)):
                if j not in I:
                    xt[j] = x[j] + (1-sum(x[k] for k in In if k not in I))/nI
 
                else: 
                    xt[j] = 0

                if xt[j] < 0:
                    I.append(j)
                    
            nI = len(x) - len(I)
            
            if len([np.real(el) for el in xt if el < 0])==len(xt):
                x = np.zeros(len(x))
            else:
                x = [xt_el for xt_el in xt]

        Lam = np.diag(xt) 
        rho_out = U.dot(Lam.dot(U.conj().T))
        
        if (round(np.trace(rho_out),5).real != 1 or round(np.trace(rho_out),5).imag != 0.0):
            print('error')
            print(rho_out)
            
        return rho_out

    def SimplexProjVec(self,a):
        '''
        Projection of 1D array 'a' onto unit simplex
        
        Input:
        a [1D array] : array to be projected
        
        Output:
        x [1D array] : projected array
        
        '''

        n=len(a)
        x=a.real.astype(float)
        x=x+self.i*a.imag.astype(float)
        I = np.linspace(0,n-1,n,dtype=int)

        for t in range(n):
            d = len(I)

            #project on V_I        
            X = sum(x[k] for k in I)
            for j in I:
                x[j] = x[j] + (1/d)*(1-X)

            N = [i for i, e in enumerate(x) if e < 0]

            if not N:
                break

            for j in N:
                x[j] = 0.0

            I = [i for i, e in enumerate(x) if e > 0]

        return x


    def rand_rho(self,lam):
        '''
        Random density matrix generator from Eliot's Github
        
        '''
        
        d = len(lam)
        rho = np.zeros((d,d),dtype=complex)
        randM = np.array([[np.random.random()*exp(self.i*2*np.pi*np.random.random()) for k in range(d)] for j in range(d)])

        Q, R = np.linalg.qr(randM)
        U = Q.dot(np.diag(np.sign(np.diag(R))))


        for k in range(d):
            psi = U[:,k]
            rho = rho + np.outer(psi,psi.conj()*lam[k])
            
        return self.SimplexProj(rho)

    def costfun(self,Nqubits,States,Clicks,rho):
        '''
        Calculates costfunction
        
        Input:
        Nqubits [int]                  : No. of qubits 
        States [array/list of arrays]  : Basis states to be used in reconstruction
        Clicks [list/array]            : Simulated measurements
        rho [array]                    : Density matrix estimate
        
        '''

        #M = np.array(States).reshape((len(States),2**Nqubits)).conj()
        Mvr = np.array([np.outer(state,state.conj()) for state in States])
        PArr = np.array([np.trace(np.outer(state,state.conj()).dot(rho)).real for state in States])
        #PArr = np.array([np.vdot(np.outer(state,state.conj()).flatten(), rho.flatten()) for state in States])

        f = 0
        if len(States)!=len(Clicks):
            print('Data and Projections are not of the same length')
                
        f = sum(Clicks[k]*mt.log(PArr[k].real) for k in range(len(States)) if PArr[k].real!=0)

        ##Method from PGD paper
        #PArrb = np.array([n/p for n,p in zip(Clicks,PArr)])
        #c = M * PArrb[:, None]

        #grad = -(c.conj().T).dot(M)
        
        ##Method from Goncalves and Eliot's Github
        grad = np.zeros(shape(Mvr[0]),dtype=complex)
        grad = sum((Clicks[k]/PArr[k])*Mvr[k] for k in range(len(States)))
            
        return -f, -grad
    
    def prob(self,x,y):
        return np.trace(np.dot(x,self.Proj(y))).real

    
class TomFuns(GenFuns):

    def __init__(self):
        self.i = complex(0,1)
        self.H=np.array([1,0])
        self.V=np.array([0,1])
        self.T = 1/np.sqrt(2)*np.add(self.H,self.V)
        self.B = 1/np.sqrt(2)*np.subtract(self.H,self.V)

        self.sb = [[self.H,self.V],[self.T,self.B]] #single basis
        
        
    def Tmat(self,x):
        '''
        Input:
        a   [list / array]  : vector for construction of T

        Returns lower triangulat form st. rho = T^\dag T is always a valid density matrix

        This method whilst general is more time consuming than just enetring the form of T. 
        Increase in optimization time by 59% (by ~100s) for N=100, Inc=8, Nqubits=2.

        '''

        x = np.asarray(x)
        self.T = np.matrix(np.zeros((int(np.sqrt(len(x))),int(np.sqrt(len(x)))),dtype=complex))
        J=0

        for j in range(int(np.sqrt(len(x)))):
            self.T[j,j]=x[j]
            J+=1

        for j in range(int(np.sqrt(len(x)))):
            for k in zip(range(j+1,int(np.sqrt(len(x)))-j+2),range(int(np.sqrt(len(x)))-j-1)):
                self.T[k[0],k[1]] = x[J]+self.i*x[J+1]
                J+=2

        return self.T

    
    def Tmatconj(self,x):
        x = np.asarray(x)
        return self.Tmat(x).getH()


    def LikeFun(self,a,Mvr,Nvr):
        '''
        Input:
        a   [list / array]  : vector for construction of T

        Output:
        fn   [array]  : function to be minimized with least squares. 

        Notes:  
        L(a) = sum_x [fn_x(a)]^2
        See 'Photonic State Tomography' notes for details

        '''
                
        a = np.asarray(a)
        ##MULTINOMIAL TRY
        # if np.prod([self.range_prod(1,x) for x in np.array(Nvr).flatten()])==mt.inf:
        #     norm=0
        # else:
        #     norm=self.range_prod(1,sum(Nvr))/np.prod([self.range_prod(1,x) for x in np.array(Nvr).flatten()])
            
        #print('LOG ARG',[np.trace(np.dot(Mvr[i],np.dot(self.Tmatconj(a),self.Tmat(a))/np.trace(np.dot(self.Tmatconj(a),self.Tmat(a))))) for i in range(len(Mvr))])
        #print('LOG',[mt.log(np.trace(np.dot(Mvr[i],np.dot(self.Tmatconj(a),self.Tmat(a))/np.trace(np.dot(self.Tmatconj(a),self.Tmat(a)))))) for i in range(len(Mvr))])
        #print('NVR', Nvr)

        self.fn = np.array([self.i*np.sqrt(2)* cp.sqrt(Nvr[i]*mt.log(np.trace(np.dot(Mvr[i],np.dot(self.Tmatconj(a),self.Tmat(a))
                                                          /np.trace(np.dot(self.Tmatconj(a),self.Tmat(a))))))) 
                            for i in range(len(Mvr))])
        #print('FN',self.fn)        
#        self.fn = np.array([np.sqrt(2)*(np.trace(np.dot(Mvr[i],np.dot(self.Tmatconj(a),self.Tmat(a)))) - Nvr[i])
#                     /np.sqrt(2*np.trace(np.dot(Mvr[i],np.dot(self.Tmatconj(a),self.Tmat(a))))) for i in range(len(Mvr))]) 
        return self.fn#.reshape(len(Mvr))


    def range_prod(self,lo,hi):
        if lo+1 < hi:
            mid = (hi+lo)//2
            return self.range_prod(lo,mid) * self.range_prod(mid+1,hi)
        if lo == hi or hi==0:
            return lo
        return lo*hi
    
    def TdagTderiv(self,i,a,k):
        a = np.asarray(a)
        dij = np.insert(np.zeros(len(a)-1),k,1)

        self.diffT = np.dot(self.Tmatconj(a),self.Tmat(dij))+np.dot(self.Tmatconj(dij),self.Tmat(a))

        return self.diffT


    def nderiv(self,i,a,k,Mvr,Nvr):
        a = np.asarray(a)
        dij = np.insert(np.zeros(len(a)-1),k,1)

        self.diffn = np.trace(np.dot(Mvr[i],np.dot(self.Tmatconj(a),self.Tmat(dij))+np.dot(self.Tmatconj(dij),self.Tmat(a))))

        return self.diffn

    
    def fderiv(self,a,Mvr,Nvr):
        '''
        Input:
        a   [list / array]  : vector for construction of T

        Output:
        difff   [array]  : vector of partial derivatives of fn

        Notes:  
        L(a) = sum_x [fn_x(a)]^2
        See 'Photonic State Tomography' notes for details

        '''

        a = np.asarray(a)
        
        self.difff = np.array([[(np.trace(np.dot(self.Tmatconj(a),self.Tmat(a)))**-1)
                                *Nvr[i]*(np.trace(np.dot(Mvr[i],np.dot(self.Tmatconj(a),self.Tmat(a))))**-1)
                                *np.trace(np.dot(Mvr[i],np.trace(np.dot(self.Tmatconj(a),self.Tmat(a)))*TdagTderiv(i,a,k) 
                                                 - np.trace(TdagTderiv(i,a,k))*np.dot(self.Tmatconj(a),self.Tmat(a))))
                                for k in range((2**self.Nqubits)**2)] for i in range(len(Mvr))])

#        self.difff = np.array([[np.sqrt(2)*((np.sqrt(2*np.trace(np.dot(Mvr[i],np.dot(self.Tmatconj(a),self.Tmat(a))))))**-1)*self.nderiv(i,a,k,Mvr,Nvr)
#                        *(1 + Nvr[i]*(np.trace(np.dot(Mvr[i],np.dot(self.Tmatconj(a),self.Tmat(a))))**-1))
#                         for k in range((2**self.Nqubits)**2)] for i in range(len(Mvr))])

        return self.difff #.reshape((len(Mvr),(2**self.Nqubits)**2))