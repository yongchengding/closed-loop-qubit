#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 14:03:44 2021

@author: jonzen
"""

from mixedstate import primerotrace, seguiendotrace, coupling, calfidelity, perturb, operatorQ, operatorP, createpointer, relaxtion
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

np.random.seed(1)
Omega = np.pi
TLS = 0.5*(np.array([[0.05*Omega,Omega],[Omega,-0.05*Omega]],dtype=complex))
tf = np.pi/Omega
dt = 0.01
steps = int(tf/dt)
tlist = np.linspace(0,tf,steps+1)
alpha = 0
beta = 0
phi = np.array([[np.cos(alpha/2)],[np.sin(alpha/2)*np.exp(-1j*beta)]],dtype='complex')
rho = np.kron(phi,np.transpose(np.conjugate(phi)))
gamma = 0
Sx = np.array([[0,1],[1,0]],dtype='complex')
Sy = np.array([[0,-1j],[1j,0]],dtype='complex')
Sz = np.array([[1,0],[0,-1]],dtype='complex')
sxlist = np.zeros(steps+1)
sylist = np.zeros(steps+1)
szlist = np.zeros(steps+1)
szlist[0] = 1
for i in range(steps):
    rho = relaxtion(Sx,gamma,TLS,rho,dt)
    #print(rho)
    sxlist[i+1] = np.real((np.trace(np.dot(rho,Sx))))
    sylist[i+1] = np.real((np.trace(np.dot(rho,Sy))))
    szlist[i+1] = np.real((np.trace(np.dot(rho,Sz))))
    
#plt.plot(tlist,szlist)

N = 20
sigma = 10
xmax = 50
dx = 1
sigma = 10
gt = 1
rho_psi = createpointer(xmax,dx,sigma)
Sxlist = np.zeros([steps+1,N])
Sylist = np.zeros([steps+1,N])
Szlist = np.zeros([steps+1,N])
Q = operatorQ(xmax,dx)
P = operatorP(Q)

for k in range(N):
    phi = np.array([[np.cos(alpha/2)],[np.sin(alpha/2)*np.exp(-1j*beta)]],dtype='complex')
    rho = np.kron(phi,np.transpose(np.conjugate(phi)))
    Szlist[0,k] = 1
    for i in range(steps):
        rho = relaxtion(Sx,gamma,TLS,rho,dt)
        totrho = np.kron(rho_psi,rho)
        totnew = coupling(totrho,gt,P,Sz)
        newpointer = primerotrace(totnew,len(totnew),len(rho_psi))
        rho = perturb(newpointer,2,totnew)[0]
        Sxlist[i+1,k] = np.real((np.trace(np.dot(rho,Sx))))
        Sylist[i+1,k] = np.real((np.trace(np.dot(rho,Sy))))
        Szlist[i+1,k] = np.real((np.trace(np.dot(rho,Sz))))

Sxmean = np.zeros(steps+1)       
Szmean = np.zeros(steps+1)
Symean = np.zeros(steps+1)
for i in range(steps+1):
    Sxmean[i] = np.average(Sxlist[i,:]) 
    Szmean[i] = np.average(Szlist[i,:])
    Symean[i] = np.average(Sylist[i,:])
    
darkred='#8B0000'
darkblue='#00008B'

fig,ax=plt.subplots(1, 1, sharey=False, sharex=False)
plt.plot(tlist,Sylist,color='darkred',alpha=0.2,linewidth=0.5)
plt.plot(tlist,Szlist,color='darkblue',alpha=0.2,linewidth=0.5)
plt.plot(tlist,Symean,color='darkred')
plt.plot(tlist,Szmean,color='darkblue')
plt.plot(tlist,sylist,color='darkred',linewidth=1,linestyle='--')
plt.plot(tlist,szlist,color='darkblue',linewidth=1,linestyle='--')
plt.xlabel(r'$t/T$',fontsize=20)
plt.ylabel(r'$\langle\hat{\sigma_y}\rangle,~\langle\hat{\sigma_z}\rangle$',fontsize=20)
ax.tick_params(axis='both', labelsize=20)
plt.title(r'$\Delta t/T=0.01,~N=20$',fontsize=20)
#fig.savefig('pipulserelaxtion.pdf',dpi=800,bbox_inches='tight',format='pdf')
np.save("sxdetuned.npy",Sxmean)
np.save("szdetuned.npy",Szmean)
np.save("sydetuned.npy",Symean)
        
        
        