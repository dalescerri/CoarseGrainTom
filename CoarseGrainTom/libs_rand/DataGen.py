
# coding: utf-8

# In[59]:


import itertools
import numpy as np
import scipy.linalg as lin
import math as mt
import cmath as cp
import operator
import time
import datetime
import six
import pickle
import itertools
from qutip import *
#from pylab import *
#from scipy import *
from scipy.optimize import least_squares
from copy import deepcopy

# import sys
# sys.path.append ('/Users/dalescerri/Tomography/Final version/FullAv/1Q')
from libs_rand import Funs as funs
from importlib import reload
reload (funs)


class DataGen():
    
    def __init__(self):
        
        self.i = complex(0,1)
        self.H=np.array([1,0])
        self.V=np.array([0,1])
        self.T = 1/np.sqrt(2)*np.add(self.H,self.V)
        self.B = 1/np.sqrt(2)*np.subtract(self.H,self.V)
        self.sb = [[self.H,self.V],[self.T,self.B]] #single basis
        
        
    def set_experiment(self,NoClicks,Slices,Nqubits,Rho,a=.98,resamp=True,n_samples=1000,
                       dist_key='uniform',scale_par=0,sort=True):
        '''
        Setting up an experiment to get fake data.
        
        Input:
        NoClicks [int]  :  Number of measurements performed 
        Slices   [int]  :  Number of segments to divide the Bloch sphere for single qubit
                           Total number of segments for N qubits is Slices^N
        Nqubits  [int]  :  Number of qubits
        Rho      [array]:  Input density matrix to be reconstructed
        
        '''
        
        self.N = NoClicks
        self.Inc = Slices
        self.Nqubits = Nqubits
        self.Sign = list(itertools.product([1,-1], repeat=self.Nqubits))
        theta_opts = {'uniform': [[[np.random.uniform(0,2*np.pi)
                                    for k in range(self.N)]
                                   for s in range(2**self.Nqubits)]
                                  for d in range(self.Nqubits)],
                      'exponential': [[[np.random.exponential(scale_par)%(2*np.pi)
                                        for k in range(self.N)]
                                       for s in range(2**self.Nqubits)]
                                      for d in range(self.Nqubits)]}
        self.tom = funs.TomFuns()
        self.tom.set_tomography(self.Nqubits)
        self.theta = theta_opts[dist_key]
        self.thetasort = np.sort(self.theta)
        self.Rho = Rho
        self.sort = sort
        self.a = a
        self.resamp = resamp
        self.n_samples = n_samples
        self.sort = sort
        self.basis_plot = np.array([qeye(2**self.Nqubits).full(),sigmax().full(),sigmay().full(),sigmaz().full()])


    def generate_states(self):
        '''
        Generate random states obtained during experiment, and the corresponding centroid states.
        Default distribution sampled: Uniform (see theta from set_experiment())
        
        Output:
        State      [array] : Array of centroid states
        StateIndiv [array] : Array of random states
        
        '''
        
        self.State = []
        self.StateIndiv = []
        
        Coords = [[[[[] for f in range(2)] for g in range(self.Inc)] for s in range(2**self.Nqubits)] for d in range(self.Nqubits)]

        for d in range(self.Nqubits):
            for s in range(2**self.Nqubits):
                for k in range(self.N):
                    for g in range(self.Inc):
                        if g*2*np.pi/self.Inc < self.thetasort[d][s][k] <= (g+1)*2*np.pi/self.Inc:
                            Coords[d][s][g][0].append(np.sin(self.thetasort[d][s][k]))
                            Coords[d][s][g][1].append(np.cos(self.thetasort[d][s][k]))

        PolyCoords = deepcopy(Coords)

        for d in range(self.Nqubits):
            for s in range(2**self.Nqubits):
                for g in range(len(PolyCoords[d][s])):
                    if Coords[d][s][g]!=[[],[]]:
                        PolyCoords[d][s][g][0].append(Coords[d][s][g][0][0])
                        PolyCoords[d][s][g][1].append(Coords[d][s][g][1][0])

        print('set coordinates')
        if self.sort:
            Centroids = [[self.tom.Centroid(PolyCoords[d][s],self.Inc) for s in range(2**self.Nqubits)] for d in range(self.Nqubits)]

            R = [[np.zeros(len(Centroids[d][s][0])) for s in range(2**self.Nqubits)] for d in range(self.Nqubits)] 
            Phi = [[np.zeros(len(Centroids[d][s][0])) for s in range(2**self.Nqubits)] for d in range(self.Nqubits)] 

            for d in range(self.Nqubits):
                for s in range(2**self.Nqubits):
                    for k in range(len(Centroids[d][s][0])):
                        if Centroids[d][s][0][k] == 0 and Centroids[d][s][1][k]==0:
                            R[d][s][k] = 0
                            Phi[d][s][k] = 0
                        else:
                            R[d][s][k] = np.sqrt(pow(Centroids[d][s][0][k],2)+pow(Centroids[d][s][1][k],2))

                            if Centroids[d][s][0][k]>=0 and Centroids[d][s][1][k]>=0:
                                Phi[d][s][k] = np.arctan(Centroids[d][s][0][k]/Centroids[d][s][1][k])

                            elif Centroids[d][s][0][k]>=0 and Centroids[d][s][1][k]<0:
                                Phi[d][s][k] = np.pi/2 + np.arctan(abs(Centroids[d][s][1][k])/Centroids[d][s][0][k])

                            elif Centroids[d][s][0][k]<0 and Centroids[d][s][1][k]<0:
                                Phi[d][s][k] = np.pi + np.arctan(abs(Centroids[d][s][0][k]/Centroids[d][s][1][k]))

                            elif Centroids[d][s][0][k]<0 and Centroids[d][s][1][k]>=0:
                                Phi[d][s][k] = 3*np.pi/2 + np.arctan(Centroids[d][s][1][k]/abs(Centroids[d][s][0][k]))

            PhiConj = [[[Phi[d][s][(k+int(self.Inc/2)) % self.Inc] for k in range(len(Centroids[d][s][0]))] 
                        for s in range(2**self.Nqubits)] for d in range(self.Nqubits)]

            for k in range(2**self.Nqubits):
                for l in range(2**self.Nqubits):
                    for Phiprod in itertools.product(*[Phi[q][k] for q in range(self.Nqubits)]):

                        Phiarr = np.array([cp.exp(self.i*sum(self.Sign[d][j]*Phiprod[j] for j in range(self.Nqubits))/2)
                                           for d in range(len(self.Sign))])

                        self.State.append(Phiarr*self.tom.MultiBasis()[0][(2**self.Nqubits)*k+l])


        #print(len(list(zip(*[self.theta[q][2**self.Nqubits-1] for q in range(self.Nqubits)])))*2**(2*self.Nqubits))
        for k in range(2**self.Nqubits):
            for l in range(2**self.Nqubits):
                for Thetaprod in zip(*[self.theta[q][k] for q in range(self.Nqubits)]):

                    Phiarrind = np.array([cp.exp(self.i*sum(self.Sign[d][j]*Thetaprod[j] for j in range(self.Nqubits))/2)
                                          for d in range(len(self.Sign))])

                    self.StateIndiv.append(Phiarrind*self.tom.MultiBasis()[0][(2**self.Nqubits)*k + l])
        print('built states')
                    
        return self.State, self.StateIndiv
                    
            
    def sort_data(self):
        '''
        Set up actual experiment. Calculate probabilities for each set of orthogonal (random) states and `roll a dice'
        to get result of experiemnt.
        Count data.
        Count data that falls within each of the centroid states.
        
        Output:
        ClickDictSort [dict] : Key (tuple): basiselem - index of corresponding basis state without rotation (ex: tensor(H,V))
                                            j         - tuple of corresponding `slice' or Bloch sphere segment
        
        '''

        if self.sort:
        
            self.ClickDict = {}
        
            for j in range((2**self.Nqubits)*(2**self.Nqubits)):
                self.ClickDict[j] = []
                
            for basissetting in range(2**self.Nqubits):
                for experiment in range(self.N):
                    prob = [np.real(np.trace(np.dot(self.Rho,self.tom.Proj(self.StateIndiv[basissetting*(2**self.Nqubits)*self.N + experiment + s*self.N]))))
                            for s in range(2**self.Nqubits)]

                    outcome = np.random.choice([basissetting*(2**self.Nqubits) + s for s in range(2**self.Nqubits)], 1, p=prob)[0]
                    self.ClickDict[outcome].append([self.theta[qubit][basissetting][experiment] for qubit in range(self.Nqubits)])
                    
                    
            self.ClickDictSort = {}
            #for setting in range(2**self.Nqubits):
            for basiselem in range((2**self.Nqubits)**2):
                for j in list(itertools.product(range(self.Inc),repeat=self.Nqubits)):
                    self.ClickDictSort[(basiselem,j)] = 0

            for basiselem in self.ClickDict:
                for angles in self.ClickDict[basiselem]:
                    for increments in list(itertools.product(range(self.Inc),repeat=self.Nqubits)):
                        if (self.tom.tupleless(tuple(np.multiply(increments,2*np.pi/self.Inc)) , tuple(angles))
                            and self.tom.tupleless(tuple(angles) , tuple(np.multiply(tuple(map(operator.add, increments, tuple(np.ones(self.Nqubits,dtype=int)))),
                                                                                2*np.pi/self.Inc)))):
                            self.ClickDictSort[(basiselem,increments)]+=1
                    
        else:
        
            self.ClickDict = np.zeros(len(self.StateIndiv))
        
            for basissetting in range(2**self.Nqubits):
                for experiment in range(self.N):
                    prob = [np.real(np.trace(np.dot(self.Rho,self.tom.Proj(self.StateIndiv[basissetting*(2**self.Nqubits)*self.N + experiment + s*self.N])))) for s in range(2**self.Nqubits)]
                    outcome = np.random.choice([basissetting*(2**self.Nqubits)*self.N + experiment + s*self.N for s in range(2**self.Nqubits)], 1, p=prob)
                    self.ClickDict[outcome]+=1
                    
            self.ClickDictSort = {}
            
        print('created measurement data')
                        
        return self.ClickDictSort, self.ClickDict
    
    def Ppost(self,M,rho_p):
        '''
        Calculates the overlap between particle p and measurement M
        
        '''
              
        Mvec = M.flatten()
        rhovec = rho_p.flatten()

        return np.trace(M.dot(rho_p))#Mvec.T.conj().dot(rhovec)
    
    def Pposn(self,particle):
        '''
        Can't use this since it can't be scaled to Nqubits>1
        '''
        
        posx = np.trace(sigmax().full().dot(particle))
        posy = np.trace(sigmay().full().dot(particle))
        posz = np.trace(sigmaz().full().dot(particle))
                
        return [posx,posy,posz]
    
    def GinibreDist(self,r=None):
        '''
        generate n_samples random samples from a ginibre dist
        '''
        
        return [rand_dm_ginibre(2**self.Nqubits, r).full() for j in range(self.n_samples)]
    
    def part_exp(self, particles):
        '''
        generate array of qeye(),sigmax(),sigmay(),sigmaz() expectation values for each particle 
        equivalent to the states used in qinfer
        '''
        
        exp_arr = []
        
        for particle in particles:
            part_exp = []
            for state in self.basis_plot:
                part_exp.append(np.trace(particle.dot(state)))
            exp_arr.append(part_exp)
            
        return exp_arr
    
    def mean_approx(self,weights,particles):
        
        mean_qinfer = np.sum(weights * np.array(self.part_exp(particles)).transpose([1,0]),axis=1)
        #print(mean_qinfer - np.array(self.part_exp([sum(w*p for w,p in zip(weights,particles))])[0]))
        
        return mean_qinfer#sum(w*p for w,p in zip(weights,particles))
    
    def cov_approx(self,weights,particles):
        
        #l = self.part_exp(particles,self.basis_plot)
        mu = self.mean_approx(weights,particles)
        #mu2 = self.part_exp([self.mean_approx(weights,particles)]).T
        
        xs = np.array(self.part_exp(particles)).transpose([1,0])
        
        cov = np.einsum('i,mi,ni', weights, xs, xs) - np.dot(mu[..., np.newaxis],mu[np.newaxis, ...]) #np.outer(mu,mu)
        #cov2 = np.einsum('i,mi,ni', weights, xs, xs) - np.dot(mu2[..., np.newaxis],mu2[np.newaxis, ...])
        #cov = sum(w*np.outer(p,p.conj()) for w,p in zip(weights,particles)) - np.outer(mu,mu.conj())
        
        # dstd = np.sqrt(np.diag(cov))
        # cov /= (np.outer(dstd, dstd))
        
        return cov#sum(w*(p.dot(p.T)) for w,p in zip(weights,particles)) - mu.dot(mu.T)
    
    def sqrtm_psd(self, A, est_error=True, check_finite=True):
        '''
        Returns the matrix square root of a positive semidefinite matrix,
        truncating negative eigenvalues.
        '''
        w, v = lin.eigh(A, check_finite=check_finite)
        mask = w <= 0
        w[mask] = 0
        np.sqrt(w, out=w)
        A_sqrt = (v * w).dot(v.conj().T)

        if est_error:
            return A_sqrt, np.linalg.norm(np.dot(A_sqrt, A_sqrt) - A, 'fro')
        else:
            return A_sqrt

    
    def smc_resample(self,weights,particles,maxiter=1000):
        
        l = np.array(self.part_exp(particles))
        mu = self.mean_approx(weights,particles)#self.part_exp([self.mean_approx(weights,particles)])[0]
        h = np.sqrt(1-self.a**2)
        n_particles = len(l)
        n_rvs = 2*len(particles[0])
        
        S, S_err = self.sqrtm_psd(self.cov_approx(weights,particles))
        S = np.real(h*S)
        new_locs = np.empty((n_particles,n_rvs),dtype=complex)
        cumsum_weights = np.cumsum(weights)
        
        idxs_to_resample = np.arange(n_particles,dtype=int)
        n_iters = 0
        js = cumsum_weights.searchsorted(np.random.random((idxs_to_resample.size,)), side = 'right')
        
        mus = np.multiply(self.a,l[js,:]) + np.multiply((1-self.a),mu)
        #mus = self.a * l[js,:] + (1 - self.a) * mu

        while idxs_to_resample.size and n_iters<maxiter:
            n_iters+=1

            new_locs[idxs_to_resample,:] = mus + np.dot(S,np.random.randn(n_rvs,mus.shape[0])).T
            ####skipped last part in qinfer
        
        ##PROJECTING ALL PTS ON SIMPLEX SEEMS TO BE CAUSING THE FLIPPING WHEN IT'S CONVERGING TO THE TRUE STATE...
        new_particles = []
        for n_loc in range(n_particles):
            
            ####removed .5* if using sqrt(.5)*basis
            new_particles.append(.5*sum(new_locs[n_loc][state]*self.basis_plot[state] for state in range(len(self.basis_plot))))
            
            if sum(new_locs[n_loc][state]**2 for state in range(1,len(self.basis_plot)))>1:
                new_particles[n_loc] = self.tom.SimplexProj(new_particles[n_loc])                

        new_weights = (n_particles**-1)*np.ones(n_particles)
        return new_particles, new_weights
    
    def update_weights(self):
        '''
        eventually move trial loop to main py script
        '''
    
        self.ClickDict = np.zeros(len(self.StateIndiv))
        self.GinDist = self.GinibreDist()
        self.GinDist2 = np.multiply(1,self.GinDist)
        self.wArr = [len(self.GinDist)**-1 for particle in range(len(self.GinDist))]
        self.n_eff = sum(w**2 for w in self.wArr)**-1
        
        for basissetting in range(2**self.Nqubits):
            for experiment in range(self.N):
                self.PostSampP = []
                if self.resamp and self.n_eff/len(self.GinDist2) < .5:
                    self.GinDist2, self.wArr = self.smc_resample(self.wArr,self.GinDist2)
                
                probs = [np.real(self.Ppost(self.tom.Proj(self.StateIndiv[basissetting*(2**self.Nqubits)*self.N + experiment + s*self.N]),self.Rho)) for s in range(2**self.Nqubits)]

                outcome = np.random.choice([basissetting*(2**self.Nqubits)*self.N + experiment + s*self.N for s in range(2**self.Nqubits)], 1, p=probs)[0]
                self.ClickDict[outcome]+=1

                for particle in range(len(self.GinDist2)):
                    self.PostSampP.append(
                        self.Ppost(self.tom.Proj(self.StateIndiv[outcome]), self.GinDist2[particle]))

                self.wArr = [p*w for p,w in zip(self.PostSampP,self.wArr)]
                self.wArr = [w/sum(self.wArr) for w in self.wArr]
                self.n_eff = sum(w**2 for w in self.wArr)**-1
                
        return .5*sum(self.mean_approx(self.wArr,self.GinDist2)[state]*self.basis_plot[state] for state in range(len(self.basis_plot)))
    
    
    def Mvr(self):
        '''
        Returns projection operators for centroid states
        
        '''

        if self.sort:
            self.States2 = self.generate_states()[0]
            self.SortedData = self.sort_data()[0]
        else:
            self.States2 = self.generate_states()[1]
            self.SortedData = self.sort_data()[1]
            
        return self.States2, [np.outer(state,np.conj(state)) for state in self.States2]
        

    def Nvr(self):
        '''
        Returns rehshaped ClickDictSort values to correspond to centroid states
        
        '''
        
        if self.sort:
            return list(np.reshape([[self.SortedData[(elem,inc)] for inc in itertools.product(range(self.Inc),repeat=self.Nqubits)] for elem in range((2**self.Nqubits)**2)] ,(1,len(self.States2)))[0])
        
        else:
            print(np.shape(self.SortedData))
            return self.SortedData
    







