import importlib
import json
import os
from threading import Thread
from tensorforce import TensorforceError, util
import numpy as np
import math
import random
from scipy.linalg import expm
from tensorforce.environments import Environment
from mixedstate import calfidelity, coupling, createpointer, dephasing, operatorP, operatorQ, perturb, primerotrace, relaxtion, seguiendotrace, sqrtm



class WEAKMEASUREMENT(Environment):
    """
    Tensorforce environment interface.
    """

    
    

    def __init__(self):
        # first two arguments, if applicable: level, visualize=False
        self.observation = None
        self.timestep = 0
        self.tf = 1
        self.Omega = 2*1.5*np.pi
        self._max_episode_timesteps = 100 #100 pulses
        self.dt = self.tf/self._max_episode_timesteps #evolution time per step
        self._states = dict(shape=(7,), type='float') #Omega q0 
        self.sz = np.array([[1,0],[0,-1]],dtype=complex)
        self.sx = np.array([[0,1],[1,0]],dtype=complex)
        self.rho = np.array([[1,0],[0,0]],dtype=complex) #initialized to |0> state
        self.xmax = 50
        self.dx = 1
        self.gt = 1
        self.sigma = 10 #the pointer is not accurate
        self.target = np.array([[0,0],[0,1]],dtype=complex) # target be |1> state
        self.gamma = 0.05
        self.escape = 0
        self.pointer = createpointer(self.xmax,self.dx,self.sigma)
        self.Q = operatorQ(self.xmax,self.dx)
        self.P = operatorP(self.Q)

        
    def __str__(self):
        return self.__class__.__name__

    def states(self):
        """
        Returns the state space specification.
        Returns:
            specification: Arbitrarily nested dictionary of state descriptions with the following
            attributes:
            <ul>
            <li><b>type</b> (<i>"bool" | "int" | "float"</i>) &ndash; state data type
            (<span style="color:#00C000"><b>default</b></span>: "float").</li>
            <li><b>shape</b> (<i>int | iter[int]</i>) &ndash; state shape
            (<span style="color:#C00000"><b>required</b></span>).</li>
            <li><b>num_states</b> (<i>int > 0</i>) &ndash; number of discrete state values
            (<span style="color:#C00000"><b>required</b></span> for type "int").</li>
            <li><b>min_value/max_value</b> (<i>float</i>) &ndash; minimum/maximum state value
            (<span style="color:#00C000"><b>optional</b></span> for type "float").</li>
            </ul>
        """
        
        return self._states

    def actions(self):
        """
        Returns the action space specification.
        Returns:
            specification: Arbitrarily nested dictionary of action descriptions with the following
            attributes:
            <ul>
            <li><b>type</b> (<i>"bool" | "int" | "float"</i>) &ndash; action data type
            (<span style="color:#C00000"><b>required</b></span>).</li>
            <li><b>shape</b> (<i>int > 0 | iter[int > 0]</i>) &ndash; action shape
            (<span style="color:#00C000"><b>default</b></span>: scalar).</li>
            <li><b>num_actions</b> (<i>int > 0</i>) &ndash; number of discrete action values
            (<span style="color:#C00000"><b>required</b></span> for type "int").</li>
            <li><b>min_value/max_value</b> (<i>float</i>) &ndash; minimum/maximum action value
            (<span style="color:#00C000"><b>optional</b></span> for type "float").</li>
            </ul>
        """
        
        return dict(shape=(1,), type='float',min_value=0.0, max_value=1.0) #output Delta

    def max_episode_timesteps(self):
        """
        Returns the maximum number of timesteps per episode.
        Returns:
            int: Maximum number of timesteps per episode.
        """
        return self._max_episode_timesteps

    def close(self):
        """
        Closes the environment.
        """
     
        
        self.environment = None

    def reset(self):
        """
        Resets the environment to start a new episode.
        Returns:
            dict[state]: Dictionary containing initial state(s) and auxiliary information.
        """
        
        self.STATES=np.array([0,(np.random.normal(loc=1, scale=self.sigma, size=None)+50)/100,0,1,0,0,0])
        #Delta q0  t=0

        self.rho = np.array([[1,0],[0,0]],dtype=complex) #initialized to |0> state
        
        self.timestep = 0
        self.escape = 0
                                                
        return self.STATES


    def execute(self, actions):
        """
        Executes the given action(s) and advances the environment by one step.
        Args:
            actions (dict[action]): Dictionary containing action(s) to be executed
                (<span style="color:#C00000"><b>required</b></span>).
        Returns:
            ((dict[state], bool | 0 | 1 | 2, float)): Dictionary containing next state(s), whether
            a terminal state is reached or 2 if the episode was aborted, and observed reward.
        """
        #detuning = actions[0]
        #delta = (2*1.5*detuning-1.5)*self.Omega # Delta in [-1.5*self.Omega,1.5*self.Omega] 
        #omega = self.Omega
        delta = 0
        omega = actions[0]*self.Omega
        '''
        Now we calculate the driving evolution: U1=sz dt/2, U2=sx dt, U3=sz dt/2
        '''
        TLS = 0.5*(np.array([[delta,omega],[omega,-delta]],dtype=complex))
        self.rho = dephasing(self.gamma,TLS,self.rho,self.dt)        
        totrho = np.kron(self.pointer,self.rho)
        
        
        '''
        Now we go for weak measurement
        '''
        totnew = coupling(totrho,self.gt,self.P,self.sz)
        newpointer = primerotrace(totnew,len(totnew),len(self.pointer))
        self.rho, index = perturb(newpointer,2,totnew)
        q0 = self.Q[index,index]

        '''
        Renormalize q0 from [-inf,inf] to [0,1]
        '''
        if q0 > 50:
            q0 = 50
        if q0 < -50:
            q0 = -50
        q0 = (q0 + 50)/100

        self.timestep += 1
        self.STATES = np.array([actions[0], q0,self.timestep/100,abs(self.rho[0,0]),abs(self.rho[0,1]),abs(self.rho[1,0]),abs(self.rho[1,1])])

        reward = 0  
        
        if abs(self.rho[1,1]) >= 0.99:
            self.escape += 1
        else:
            self.escape = 0
        if self.escape == 1:
            terminal = 1
            reward += 1000
        elif self.timestep == self._max_episode_timesteps:
            terminal = 1
            reward += (abs(self.rho[1,1]) - 1)
            if abs(self.rho[0,0]) >= 0.05:
                reward -= 100
        else:
            terminal = 0
            reward += (abs(self.rho[1,1]) - 1)
        
        

        '''
        if self.timestep < self._max_episode_timesteps:
            terminal = 0
            #reward -= abs(self.STATES[0]-self.STATES[2])
        else:
            terminal = 1
            fidelity = calfidelity(self.target,self.rho)
            print(fidelity)
            reward -= np.log(1-fidelity)
        '''
       
        #pretrain
        '''
        if self.timestep < self._max_episode_timesteps:
            reward -= abs(actions[0]-self.timestep/self._max_episode_timesteps)
            terminal = 0
        else:
            reward -= abs(actions[0]-self.timestep/self._max_episode_timesteps)
            terminal = 1
        '''
        return self.STATES, terminal, reward
            