#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 19:53:05 2021
Weak measurement on mixed states
@author: jonzen
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, sqrtm


def primerotrace(totrho,totdim,firstdim):
    '''
    we assume a total system reads \rho_T = \rho_A\otimes\rho_B
    this retrieves \rho_a by a partial trace
    Return: rho_A as a complex matrix
    '''
    I_A = np.eye(firstdim,dtype=complex)
    rho_A = np.zeros([firstdim,firstdim],dtype=complex)
    seconddim = int(totdim/firstdim)
    for i in range(seconddim):
        b = np.zeros([seconddim,1],dtype=complex)
        b[i] = 1
        Ul = np.kron(I_A,np.transpose(b))
        Ur = np.kron(I_A,b)
        rho_A += np.dot(np.dot(Ul,totrho),Ur)
    return rho_A
   
def seguiendotrace(totrho,totdim,seconddim):
    '''
    we assume a total system reads \rho_T = \rho_A\otimes\rho_B
    this retrieves \rho_B by a partial trace
    Return: rho_B as a complex matrix
    '''
    I_B = np.eye(seconddim,dtype=complex)
    rho_B = np.zeros([seconddim,seconddim],dtype=complex)
    firstdim = int(totdim/seconddim)
    for i in range(firstdim):
        b = np.zeros([firstdim,1],dtype=complex)
        b[i] = 1
        Ul = np.kron(np.conjugate(np.transpose(b)),I_B)
        Ur = np.kron(b,I_B)
        rho_B += np.dot(np.dot(Ul,totrho),Ur)
    return rho_B

def coupling(totrho,gt,operator_A,operator_B):
    '''
    we evovle the interaction Hamiltonian H = g(t)\hat{A}\otimes\hat{B} for t second
    gt is the pulse area, in practice we assume it is a delta function
    \hat{A} is the pointer and \hat{B} is the system to be weakly measured 
    '''
    Ul = expm(-1j*gt*np.kron(operator_A,operator_B))
    Ur = expm(1j*gt*np.kron(operator_A,operator_B))
    rho = np.dot(np.dot(Ul,totrho),Ur)
    return rho

def calfidelity(rho_A,rho_B):
    F = abs(np.trace(sqrtm(np.dot(np.dot(sqrtm(rho_A),rho_B),sqrtm(rho_A)))))**2
    if F >= 1:
        F  = 1
    return F
    

def perturb(rho_A,Bdim,totrho):
    '''
    
    '''
    rho_B = seguiendotrace(totrho,len(totrho),Bdim)
    Adim = len(rho_A)
    indexlist = np.linspace(0,Adim-1,Adim,dtype=int)
    plist = abs(np.diag(rho_A))
    index = np.random.choice(indexlist,1,p=plist)[0]
    b = np.zeros([Adim,1],dtype=complex)
    b[index] = 1
    proj = np.kron(b,np.transpose(b))
    IB = np.eye(Bdim,dtype=complex)
    projtot = np.kron(proj,IB)
    T = np.dot(np.dot(projtot,totrho),projtot)
    newB =  seguiendotrace(T,len(T),len(rho_B))
    newB = newB/np.trace(newB)
    return newB, index
    
def operatorQ(xmax,dx):
    return np.diag(np.linspace(-xmax,xmax,int(2*xmax/dx+1),dtype=complex))

def operatorP(operatorQ):
    #deltax = (operatorQ[1,1]-operatorQ[0,0])/(-operatorQ[0,0])
    P = np.zeros([len(operatorQ),len(operatorQ)],dtype=complex)
    for i in range(len(operatorQ)-1):
        P[i,i+1] = 1
        P[i+1,i] = -1
    P[len(operatorQ)-1,0] = 1
    P[0,len(operatorQ)-1] = -1
    dx = operatorQ[1,1]-operatorQ[0,0]
    P *= (-1j/(2*dx))
    #P *= (-1j/(2*deltax))
    return P

def createpointer(xmax,dx,sigma):
    q = np.linspace(-xmax,xmax,int(2*xmax/dx+1))
    psi = np.zeros([len(q),1],dtype=complex)
    for i in range(len(q)):
        psi[i] = (2*np.pi*sigma**2)**(1/4)*np.exp(-(q[i]**2/(4*sigma**2)))
    rho = np.kron(psi,np.conjugate(np.transpose(psi)))
    rho = rho/np.trace(rho)
    return rho

def relaxtion(A,gamma,H,rho,dt):
    '''
    
    '''
    C = np.sqrt(gamma)*A
    Cdag = np.transpose(np.conjugate(C))
    
    ddt = dt/100
    for i in range(100):
        master_term = -1j*(np.dot(H,rho)-np.dot(rho,H))
        lind_term = 0.5*(2*np.dot(np.dot(C,rho),Cdag)-np.dot(np.dot(rho,Cdag),C)-np.dot(np.dot(Cdag,C),rho))
        lhs = master_term + lind_term
        rho = rho + lhs*ddt        
    return rho

def dephasing(Gamma,H,rho,dt):
    ddt = dt/100
    for i in range(100):
        master_term = -1j*(np.dot(H,rho)-np.dot(rho,H))
        lindterm = -Gamma*(rho - np.diag(np.diag(rho)))        
        #lindterm = 0
        lhs = master_term + lindterm
        rho = rho + lhs*ddt
    return rho

if __name__=="__main__":
    '''
    alpha1 = np.pi/2
    phi1 = np.array([[np.cos(alpha1/2)],[np.sin(alpha1/2)]],dtype=complex)
    rho_A = np.kron(phi1,np.conjugate(np.transpose(phi1)))
    alpha2 = np.pi/3
    phi2 = np.array([[np.cos(alpha2/2)],[np.sin(alpha2/2)]],dtype=complex)
    rho_B = np.kron(phi2,np.conjugate(np.transpose(phi2)))
    Sy = np.array([[0,-1j],[1j,0]],dtype=complex)
    Sz = np.array([[1,0],[0,-1]],dtype=complex)
    '''
    
    '''
    rho_T = np.kron(rho_A,rho_B)
    gt = 100
    new_T = coupling(rho_T,gt,Sy,Sz)
    new_A = primerotrace(new_T,len(new_T),len(rho_A))
    new_B = seguiendotrace(new_T,len(new_T),len(rho_B))
    #print(calfidelity(new_A,rho_A))
    #print(calfidelity(new_B,rho_B))
    
    proj = np.array([[0,0],[0,1]],dtype='complex')
    I22 = np.array([[1,0],[0,1]],dtype='complex')
    projtot = np.kron(proj,I22)
    T = np.dot(np.dot(projtot,new_T),projtot)
    retB =  seguiendotrace(T,len(T),len(rho_B))
    retB = retB/np.trace(retB)
    '''
    '''ds
    alpha = 5*np.pi/6
    beta = np.pi/6
    phi = np.array([[np.cos(alpha/2)],[np.sin(alpha/2)*np.exp(-1j*beta)]],dtype=complex)
    
    alpha1 = np.pi/4
    phi1 = np.array([[np.cos(alpha1/2)],[np.sin(alpha1/2)*np.exp(-1j*beta)]],dtype=complex)
    
    xmax = 50
    dx = 0.1
    sigma = 10
    Q = operatorQ(xmax,0.1)
    P = operatorP(Q)
    Sz = np.array([[1,0],[0,-1]],dtype=complex)
    rho_psi = createpointer(xmax,dx,sigma)
    rho_phi = 0.7*np.kron(phi,np.transpose(np.conjugate(phi)))+0.3*np.kron(phi,np.transpose(np.conjugate(phi1)))
    totrho = np.kron(rho_psi,rho_phi)
    gt = 1
    totnew = coupling(totrho,gt,P,Sz)
    psinew = primerotrace(totnew,len(totnew),len(rho_psi))
    
    #fig,ax=plt.subplots(1, 1, sharey=False, sharex=False)
    #plt.plot(np.diag(abs(rho_psi)))
    #plt.plot(np.diag(abs(psinew)))
    print(np.trace(np.dot(rho_psi,Q)))
    print(np.trace(np.dot(psinew,Q)))
    print(np.trace(np.dot(rho_phi,Sz)))
    #print(np.cos(alpha))
    '''
    
    
    
    
    