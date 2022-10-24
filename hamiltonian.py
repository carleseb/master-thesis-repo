#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 09:25:09 2022

@author: ceboncompte
"""

from qutip import *
import numpy as np

def qeye_rec(spins): # Called in hheis_general
    
    """Recursive function for qutip 2-dim identity tensor product."""
    
    if spins == 1:
        eyecomponent = qeye(2)
    else:
        eyecomponent = tensor(qeye(2), qeye_rec(spins-1))
    return eyecomponent

def hheis_general(Jij_vector, spins, B):
    
    """It returns a chain Heisenberg Hamiltonian of an arbitrary dimension (arbitrary number of spins)."""
    
    CORE1 = tensor(sigmax(), sigmax()) + tensor(sigmay(), sigmay()) + tensor(sigmaz(), sigmaz()) - tensor(qeye(2), qeye(2))
    CORE2 = tensor(sigmaz(), qeye(2)) + tensor(qeye(2), sigmaz())
    if spins == 2:
        hheis = (Jij_vector[0]/4)*CORE1 + B*CORE2
    else:
        qeye_with_CORE1 = qeye_rec(spins-2)
        qeye_with_sigma = qeye_rec(spins-1)
        removeone = Jij_vector[:-1]
        hheis = tensor(hheis_general(removeone, spins-1, B), qeye(2))
        hheis += (Jij_vector[-1]/4)*(tensor(qeye_with_CORE1, CORE1))
        hheis += B*(tensor(qeye_with_sigma, sigmaz()))
    return hheis

def heisenberg_hamiltonian_4(J12, J23, J34, B):

    """Returns a 4-spins Heisenberg Hamiltonian in matrix form in the computational basis as a quantum object given the input parameters."""
    
    term12 = (J12/4)*(tensor(sigmax(), sigmax(), qeye(2), qeye(2)) + tensor(sigmay(), sigmay(), qeye(2), qeye(2)) + tensor(sigmaz(), sigmaz(), qeye(2), qeye(2)) - tensor(qeye(2), qeye(2), qeye(2), qeye(2)))
    term23 = (J23/4)*(tensor(qeye(2), sigmax(), sigmax(), qeye(2)) + tensor(qeye(2), sigmay(), sigmay(), qeye(2)) + tensor(qeye(2), sigmaz(), sigmaz(), qeye(2)) - tensor(qeye(2), qeye(2), qeye(2), qeye(2)))
    term34 = (J34/4)*(tensor(qeye(2), qeye(2), sigmax(), sigmax()) + tensor(qeye(2), qeye(2), sigmay(), sigmay()) + tensor(qeye(2), qeye(2), sigmaz(), sigmaz()) - tensor(qeye(2), qeye(2), qeye(2), qeye(2)))
    mfieldterm = B*(tensor(sigmaz(), qeye(2), qeye(2), qeye(2)) + tensor(qeye(2), sigmaz(), qeye(2), qeye(2)) + tensor(qeye(2), qeye(2), sigmaz(), qeye(2)) + tensor(qeye(2), qeye(2), qeye(2), sigmaz()))
    
    return term12 + term23 + term34 + mfieldterm

def heisenberg_hamiltonian_3(J12, J23, B):

    """Returns a 3-spins Heisenberg Hamiltonian in matrix form in the computational basis as a quantum object given the input parameters"""
    
    term12 = (J12/4)*(tensor(sigmax(), sigmax(), qeye(2)) + tensor(sigmay(), sigmay(), qeye(2)) + tensor(sigmaz(), sigmaz(), qeye(2)) - tensor(qeye(2), qeye(2), qeye(2)))
    term23 = (J23/4)*(tensor(qeye(2), sigmax(), sigmax()) + tensor(qeye(2), sigmay(), sigmay()) + tensor(qeye(2), sigmaz(), sigmaz()) - tensor(qeye(2), qeye(2), qeye(2)))
    mfieldterm = B*(tensor(sigmaz(), qeye(2), qeye(2)) + tensor(qeye(2), sigmaz(), qeye(2)) + tensor(qeye(2), qeye(2), sigmaz()))
    return term12 + term23 + mfieldterm