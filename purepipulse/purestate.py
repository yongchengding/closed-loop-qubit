#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 14:47:49 2021
Weak measurement on pure states (Python version of the weak measurement in PRL124)
@author: jonzen
"""
import numpy as np

def expectation(phi,operator):
    '''
    phi is a complex 2d normalized ket np.complex
    operator is \hat{A} to be weakly measured
    return: scalar expectation <Phi|A|Phi>
    '''
    return np.real(np.dot(np.dot(np.conjugate(np.transpose(phi)),operator),phi))[0][0]

def measurepointer(expectation,sigma):
    '''
    expectation is a scalar = <Phi|A|Phi>
    sigma is the standard derivation of the pointer's wave function
    return: scalar q0 as the weak value
    '''
    return np.random.normal(loc=expectation, scale=sigma, size=None)

def perturbed(phi,q0,sigma):
    '''
    phi is the system's wave function before measurement
    q0 is the weak value
    sigma is the standard derivation of the pointer's wave function
    ATTENTION!!!!!! the perturbation is calculated for weakly measuring Sz
    return: the system's wave function complex 2d normalized ket np.complex after measurement
    '''
    phi1 = phi[0][0]*np.exp(-((q0-1)**2)/(4*sigma**2))
    phi2 = phi[1][0]*np.exp(-((q0+1)**2)/(4*sigma**2))
    norm=1/np.sqrt((abs(phi1)**2+abs(phi2)**2))
    phi1 *= norm
    phi2 *= norm
    newphi = np.array([[phi1],[phi2]],dtype='complex')
    return newphi

if __name__=="__main__":
    sz = np.array([[1,0],[0,-1]],dtype='complex')
    alpha = 1.83
    beta=0.1
    phi = np.array([[np.cos(alpha/2)],[np.sin(alpha/2)*np.exp(-1j*beta)]],dtype='complex')
    sigma = 10
    #print(expectation(phi,sz))
    #print(np.cos(alpha))
    expect_sz = expectation(phi,sz)
    #print(expect_sz)
    #print(measurepointer(expect_sz,sigma))
    q0 = measurepointer(expect_sz,sigma)
    newphi = perturbed(phi,q0,sigma)
    normalized = abs(np.dot(np.conjugate(np.transpose(newphi)),newphi))[0][0]
    fidelity = (abs(np.dot(np.conjugate(np.transpose(newphi)),phi))[0][0])**2
    print('the wave function before measurement:\n',phi)    
    print('the expectation of sz=',expect_sz)
    print('the pointer is not accurate, sigma=',sigma)
    print('the weak value q0=',q0)
    print('the wave function after measurement:\n',newphi)
    print('check if the newphi is normalized=',normalized)
    print('fidelity after perturbation=',fidelity)

