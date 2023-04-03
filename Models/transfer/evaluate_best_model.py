from tensorforce.agents import TRPOAgent
from tensorforce.agents import PPOAgent
from tensorforce.agents import Agent
import RELAXENV
import numpy as np
from tensorforce.execution import Runner
import pickle
import os
from scipy.linalg import expm
import matplotlib.pyplot as plt
from mixedstate import calfidelity, coupling, createpointer, dephasing, operatorP, operatorQ, perturb, primerotrace, relaxtion, seguiendotrace, sqrtm

os.environ['KMP_DUPLICATE_LIB_OK']='True'
#os.environ['KMP_DUPLICATE_LIB_OK']='True'




WEAKMEASUREMENT=RELAXENV.WEAKMEASUREMENT()

#hidden layer
network_spec=[
        dict(type='dense',size=64,activation='relu'),
        dict(type='dense',size=64,activation='relu'),
        dict(type='dense',size=64,activation='relu'),
        ]



agent=PPOAgent(
        states=WEAKMEASUREMENT.states(),
        actions=WEAKMEASUREMENT.actions(),
        network=network_spec,
        max_episode_timesteps=WEAKMEASUREMENT.max_episode_timesteps(),
        
        batch_size=20,
        )

np.random.seed(1)

agent.restore(directory='best-model',filename='agent-709989')

N = 20
tf = 1
dt = 0.01
sigma = 10
Omega = 2*1.5*np.pi
steps = int(tf/dt)
Sxlist = np.zeros([steps,N])
Sylist = np.zeros([steps,N])
Szlist = np.zeros([steps,N])
tlist = np.linspace(dt,tf,steps)
detuninglist = np.zeros([steps,N])
omegalist = np.zeros([steps,N])
q0list = np.zeros([steps,N])
fidelitylist = np.zeros(N)
Sx = np.array([[0,1],[1,0]],dtype=complex)
Sy = np.array([[0,-1j],[1j,0]],dtype=complex)
Sz = np.array([[1,0],[0,-1]],dtype=complex)
target = WEAKMEASUREMENT.target # target be |1> state


temp = []

for k in range(N):
    STATES = WEAKMEASUREMENT.reset()
    terminal = False
    i = 0
    #phi = np.array([[1],[0]],dtype=complex) #initialized to |0> state
    while not terminal:
        actions = agent.act(states=STATES, evaluation=True)
        STATES, terminal, reward = WEAKMEASUREMENT.execute(actions=actions)
        rho = WEAKMEASUREMENT.rho
        Sxlist[i,k] = np.trace(np.dot(rho,Sx))    
        Sylist[i,k] = np.trace(np.dot(rho,Sy))  
        Szlist[i,k] = np.trace(np.dot(rho,Sz))  
        detuning = actions[0]
        #detuninglist[i,k] = (2*1.5*detuning-1.5)*Omega # Delta in [-2*self.Omega,2*self.Omega] 
        detuninglist[i,k] = actions[0]*Omega
        q0list[i,k] = STATES[1]*100-50
        
        i += 1
    if terminal:
        print(i)
        for l in range(i,steps):
            Sxlist[l,k] = Sxlist[i-1,k]
            Sylist[l,k] = Sylist[i-1,k]
            Szlist[l,k] = Szlist[i-1,k]
    fidelitylist[k] = calfidelity(rho,target)

Sxmean = np.zeros(steps)        
Szmean = np.zeros(steps)
Symean = np.zeros(steps)
Qmean = np.zeros(steps)
for i in range(steps):
    Szmean[i] = np.average(Szlist[i,:])
    Symean[i] = np.average(Sylist[i,:])
    Sxmean[i] = np.average(Sxlist[i,:])
    Qmean[i] = np.average(q0list[i,:])        
        
darkgreen='#294537'         
darkred='#8B0000'
darkblue='#00008B'

baselinez = np.load("Sz_DRL_before_transfer.npy")
baseliney = np.load("Sy_DRL_before_transfer.npy")
baselinex = np.load("Sx_DRL_before_transfer.npy")


fig,ax=plt.subplots(1, 1, sharey=False, sharex=False)
plt.plot(tlist,Sxlist,color='darkgreen',alpha=0.2,linewidth=0.5)
plt.plot(tlist,Sylist,color='darkred',alpha=0.2,linewidth=0.5)
plt.plot(tlist,Szlist,color='darkblue',alpha=0.2,linewidth=0.5)
plt.plot(tlist,Symean,color='darkred',label=r'$\overline{\langle\hat{\sigma_y}\rangle}$, NE, NM')
plt.plot(tlist,Szmean,color='darkblue',label=r'$\overline{\langle\hat{\sigma_z}\rangle}$, NE, NM')
plt.plot(tlist,Sxmean,color='darkgreen',label=r'$\overline{\langle\hat{\sigma_x}\rangle}$, NE, NM')
plt.plot(tlist,baselinex,color='darkgreen',linewidth=1,linestyle='--',label=r'$\overline{\langle\hat{\sigma_y}\rangle}$, NE, OM')
plt.plot(tlist,baseliney,color='darkred',linewidth=1,linestyle='--',label=r'$\overline{\langle\hat{\sigma_z}\rangle}$, NE, OM')
plt.plot(tlist,baselinez,color='darkblue',linewidth=1,linestyle='--',label=r'$\overline{\langle\hat{\sigma_x}\rangle}$, NE, OM')
plt.xlabel(r'$t/T$',fontsize=20)
plt.ylabel(r'$\langle\hat{\sigma_x}\rangle,~\langle\hat{\sigma_y}\rangle,~\langle\hat{\sigma_z}\rangle$',fontsize=20)
ax.tick_params(axis='both', labelsize=20)
plt.title(r'$\Delta t/T=0.01,~N=20$',fontsize=20)
plt.legend(loc='upper right',ncol=2,frameon=False)
fig.savefig('rhodrl_transfered.pdf',dpi=800,bbox_inches='tight',format='pdf')

fig,ax=plt.subplots(1, 1, sharey=False, sharex=False)
plt.step(np.insert(tlist, 0, 0, axis=0),np.insert(detuninglist, 0, detuninglist[0], axis=0)/np.pi,linewidth=0.5)
plt.xlabel(r'$t/T$',fontsize=20)
plt.ylabel(r'$\Omega/\pi$',fontsize=20)
ax.tick_params(axis='both', labelsize=20)
fig.savefig('omegadrl_transfered.pdf',dpi=800,bbox_inches='tight',format='pdf')

