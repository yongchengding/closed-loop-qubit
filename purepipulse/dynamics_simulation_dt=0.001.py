#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 15:46:50 2021

@author: jonzen
"""

from purestate import expectation, measurepointer, perturbed
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm


#from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

'''
system parameter, simpliest case, pi-pulse
'''


np.random.seed(1)
Omega = np.pi/2
TLS = np.array([[0,Omega],[Omega,0]],dtype=complex)
tf = np.pi/(2*Omega)
dt = 0.001
steps = int(tf/dt)
tlist = np.linspace(0,tf,steps+1)
alpha = 0
beta = 0
phi = np.array([[np.cos(alpha/2)],[np.sin(alpha/2)*np.exp(-1j*beta)]],dtype='complex')
#Sx = np.array([[0,1],[1,0]],dtype='complex')
Sy = np.array([[0,-1j],[1j,0]],dtype='complex')
Sz = np.array([[1,0],[0,-1]],dtype='complex')
#sxlist = np.zeros(steps)
sylist = np.zeros(steps+1)
szlist = np.zeros(steps+1)
szlist[0] = 1
U = expm(-1j*TLS*dt)
for i in range(steps):
    phi = np.dot(U,phi)
    #sxlist[i] = np.real(np.dot(np.dot(np.conjugate(np.transpose(phi)),Sx),phi))[0][0]
    sylist[i+1] = np.real(np.dot(np.dot(np.conjugate(np.transpose(phi)),Sy),phi))[0][0]
    szlist[i+1] = np.real(np.dot(np.dot(np.conjugate(np.transpose(phi)),Sz),phi))[0][0]

#fig,ax=plt.subplots(1, 1, sharey=False, sharex=False)

#plt.plot(tlist,sylist,color='b')
#plt.plot(tlist,szlist,color='k')



'''
continuous measurement
'''
phi = np.array([[np.cos(alpha/2)],[np.sin(alpha/2)*np.exp(-1j*beta)]],dtype='complex')
sigma = 10
U = expm(-1j*TLS*dt)
N = 20
#Sxlist = np.zeros([steps,N])
Sylist = np.zeros([steps+1,N])
Szlist = np.zeros([steps+1,N])
Qlist = np.zeros([steps+1,N])
for k in range(N):
    phi = np.array([[np.cos(alpha/2)],[np.sin(alpha/2)*np.exp(-1j*beta)]],dtype='complex')
    Szlist[0,k] = 1
    for i in range(steps):
        phi = np.dot(U,phi)
        expect_sz = expectation(phi,Sz)
        q0 = measurepointer(expect_sz,sigma)    
        phi = perturbed(phi,q0,sigma)
        #Sxlist[i,k] = np.real(np.dot(np.dot(np.conjugate(np.transpose(phi)),Sx),phi))[0][0]
        Sylist[i+1,k] = np.real(np.dot(np.dot(np.conjugate(np.transpose(phi)),Sy),phi))[0][0]
        Szlist[i+1,k] = np.real(np.dot(np.dot(np.conjugate(np.transpose(phi)),Sz),phi))[0][0]
        Qlist[i+1,k] = q0

Szmean = np.zeros(steps+1)
Symean = np.zeros(steps+1)
Qmean = np.zeros(steps+1)
Qstd = np.zeros(steps+1)
for i in range(steps+1):
    Szmean[i] = np.average(Szlist[i,:])
    Symean[i] = np.average(Sylist[i,:])
    Qmean[i] = np.average(Qlist[i,:])
    Qstd[i] = np.std(Qlist[i,:])

darkred='#8B0000'
darkblue='#00008B'

fig,ax=plt.subplots(1, 1, sharey=False, sharex=False)
plt.plot(tlist,Sylist,color='darkred',alpha=0.2,linewidth=0.5)
plt.plot(tlist,Szlist,color='darkblue',alpha=0.2,linewidth=0.5)
plt.plot(tlist,Symean,color='darkred',label=r'$\overline{\langle\hat{\sigma_y}\rangle}$, CL')
plt.plot(tlist,Szmean,color='darkblue',label=r'$\overline{\langle\hat{\sigma_y}\rangle}$, CL')
plt.plot(tlist,sylist,color='darkred',linewidth=1,linestyle='--',label=r'$\langle\hat{\sigma_y}\rangle$, OL')
plt.plot(tlist,szlist,color='darkblue',linewidth=1,linestyle='--',label=r'$\langle\hat{\sigma_y}\rangle$, OL')
plt.xlabel(r'$t/T$',fontsize=20)
plt.ylabel(r'$\langle\hat{\sigma_y}\rangle,~\langle\hat{\sigma_z}\rangle$',fontsize=20)
ax.tick_params(axis='both', labelsize=20)
plt.legend(loc='upper right',frameon=False)
plt.title(r'$\Delta t/T=0.001,~N=20$',fontsize=20)
fig.savefig('pipulsesigma0001.pdf',dpi=800,bbox_inches='tight',format='pdf')
#np.save("szmean.npy",Szmean)
#np.save("symean.npy",Symean)

fig,ax=plt.subplots(1, 1, sharey=False, sharex=False)
plt.plot(tlist[1:len(tlist)],Qlist[1:len(Qlist)],color='k',alpha=0.2,linewidth=0.5)
plt.plot(tlist[1:len(tlist)],Qmean[1:len(Qmean)],color='gray',label='$\overline{q_0}$')
plt.plot(tlist[1:len(tlist)],Qstd[1:len(Qmean)],color=darkblue,linestyle='--',label='$\sigma(q_0)$')
plt.legend(loc='upper right',frameon=False)
plt.xlim([-0.05,1.05])
plt.xlabel(r'$t/T$',fontsize=20)
plt.ylabel(r'$q_0$, $\sigma(q_0)$',fontsize=20)
ax.tick_params(axis='both', labelsize=20)
plt.title(r'$\Delta t/T=0.001,~N=20$',fontsize=20)
fig.savefig('pipulseq0001.pdf',dpi=800,bbox_inches='tight',format='pdf')


    
    