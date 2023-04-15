#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 09:25:09 2022

@author: ceboncompte
"""

from qutip import *
import numpy as np

# Note: all B values here correspond to the energy value B/2 = Btilda = g*mu*Bext therefore inputs when using the fucntion
# have to be the value Btilda = g*mu*Bext, with Bext the extrernal magnetic field.

def qeye_rec(spins): # Called in hheis_general
    
    """Recursive function for qutip 2-dim identity tensor product."""
    
    if spins == 1:
        eyecomponent = qeye(2)
    else:
        eyecomponent = tensor(qeye(2), qeye_rec(spins-1))
    return eyecomponent

def hheis_general(Jij_vector, spins, B): #Jij vector is [J12,J23,J34,...], use only if spins >=2
    
    """It returns a chain (first nearest neighbours) Heisenberg Hamiltonian of an arbitrary dimension (arbitrary number of spins)."""

    core1 = tensor(sigmax(), sigmax()) + tensor(sigmay(), sigmay()) + tensor(sigmaz(), sigmaz()) - tensor(qeye(2), qeye(2))
    core2 = tensor(sigmaz(), qeye(2)) + tensor(qeye(2), sigmaz())
    if spins == 2:
        hheis = (Jij_vector[0]/4)*core1 + (B/2)*core2
    else:
        qeye_with_core1 = qeye_rec(spins-2)
        qeye_with_sigma = qeye_rec(spins-1)
        removeone = Jij_vector[:-1]
        hheis = tensor(hheis_general(removeone, spins-1, B), qeye(2))
        hheis += (Jij_vector[-1]/4)*(tensor(qeye_with_core1, core1))
        hheis += (B/2)*(tensor(qeye_with_sigma, sigmaz()))
    return hheis

def ladder_exchanges(Jij_vector, spins): #Jij vector is [J13,J24,J35,...], use only if spins >=4
    
    """It returns a second nearest neigbours Heisenberg Hamiltonian of an arbitrary dimension."""
    
    core3 = tensor(sigmax(), qeye(2), sigmax()) + tensor(sigmay(), qeye(2), sigmay()) + tensor(sigmaz(), qeye(2), sigmaz()) - tensor(qeye(2), qeye(2), qeye(2))
    if spins == 3:
        hladder = (Jij_vector[0]/4)*core3
    else:
        qeye_with_core3 = qeye_rec(spins-3)
        removeone = Jij_vector[:-1]
        hladder = tensor(ladder_exchanges(removeone, spins-1), qeye(2))
        hladder += (Jij_vector[-1]/4)*(tensor(qeye_with_core3, core3))
    return hladder

def chain_bc(Jbc, spins): # use only if spins >=3
    
    """Returns the periodic boundary condition term (JN1 = Jbc) for a spin chain of an arbitrary dimension."""
    
    qeye_with_none = qeye_rec(spins)
    qeye_with_core1_bc = qeye_rec(spins-2)
    core1_bc = tensor(sigmax(), qeye_with_core1_bc, sigmax()) + tensor(sigmay(), qeye_with_core1_bc, sigmay()) + tensor(sigmaz(), qeye_with_core1_bc, sigmaz()) - qeye_with_none
    chain_bc_term = (Jbc/4)*core1_bc
    return chain_bc_term

def ladder_bc(Jw1, Jw2, spins): # use only if spins >=5
    
    """Returns the periodic boundary condition terms for a spin ladder of an arbitrary dimension."""
    
    qeye_with_none = qeye_rec(spins)
    qeye_with_core_bc = qeye_rec(spins-3)
    core2_bc = tensor(sigmax(), qeye_with_core_bc, sigmax(), qeye(2)) + tensor(sigmay(), qeye_with_core_bc, sigmay(), qeye(2)) + tensor(sigmaz(), qeye_with_core_bc, sigmaz(), qeye(2)) - qeye_with_none
    core3_bc = tensor(qeye(2), sigmax(), qeye_with_core_bc, sigmax()) + tensor(qeye(2), sigmay(), qeye_with_core_bc, sigmay()) + tensor(qeye(2), sigmaz(), qeye_with_core_bc, sigmaz()) - qeye_with_none
    ladder_bc_term = (Jw1/4)*core2_bc + (Jw2/4)*core3_bc
    return ladder_bc_term

def heisenberg_hamiltonian_4(J12, J23, J34, Btilda):

    """Returns a 4-spins Heisenberg Hamiltonian in matrix form in the computational basis as a quantum object given the input parameters."""
    
    B = (1/2)*Btilda
    term12 = (J12/4)*(tensor(sigmax(), sigmax(), qeye(2), qeye(2)) + tensor(sigmay(), sigmay(), qeye(2), qeye(2)) + tensor(sigmaz(), sigmaz(), qeye(2), qeye(2)) - tensor(qeye(2), qeye(2), qeye(2), qeye(2)))
    term23 = (J23/4)*(tensor(qeye(2), sigmax(), sigmax(), qeye(2)) + tensor(qeye(2), sigmay(), sigmay(), qeye(2)) + tensor(qeye(2), sigmaz(), sigmaz(), qeye(2)) - tensor(qeye(2), qeye(2), qeye(2), qeye(2)))
    term34 = (J34/4)*(tensor(qeye(2), qeye(2), sigmax(), sigmax()) + tensor(qeye(2), qeye(2), sigmay(), sigmay()) + tensor(qeye(2), qeye(2), sigmaz(), sigmaz()) - tensor(qeye(2), qeye(2), qeye(2), qeye(2)))
    mfieldterm = B*(tensor(sigmaz(), qeye(2), qeye(2), qeye(2)) + tensor(qeye(2), sigmaz(), qeye(2), qeye(2)) + tensor(qeye(2), qeye(2), sigmaz(), qeye(2)) + tensor(qeye(2), qeye(2), qeye(2), sigmaz()))
    
    return term12 + term23 + term34 + mfieldterm

def heisenberg_hamiltonian_3(J12, J23, Btilda):

    """Returns a 3-spins Heisenberg Hamiltonian in matrix form in the computational basis as a quantum object given the input parameters"""
    
    B = (1/2)*Btilda
    term12 = (J12/4)*(tensor(sigmax(), sigmax(), qeye(2)) + tensor(sigmay(), sigmay(), qeye(2)) + tensor(sigmaz(), sigmaz(), qeye(2)) - tensor(qeye(2), qeye(2), qeye(2)))
    term23 = (J23/4)*(tensor(qeye(2), sigmax(), sigmax()) + tensor(qeye(2), sigmay(), sigmay()) + tensor(qeye(2), sigmaz(), sigmaz()) - tensor(qeye(2), qeye(2), qeye(2)))
    mfieldterm = B*(tensor(sigmaz(), qeye(2), qeye(2)) + tensor(qeye(2), sigmaz(), qeye(2)) + tensor(qeye(2), qeye(2), sigmaz()))
    return term12 + term23 + mfieldterm

def heisenberg_ladder_4(J12, J23, J34, Btilda, J13, J24):

    """Returns a 4-spins Heisenberg Hamiltonian in matrix form in the computational basis as a quantum object given the input parameters."""
    
    B = (1/2)*Btilda
    term12 = (J12/4)*(tensor(sigmax(), sigmax(), qeye(2), qeye(2)) + tensor(sigmay(), sigmay(), qeye(2), qeye(2)) + tensor(sigmaz(), sigmaz(), qeye(2), qeye(2)) - tensor(qeye(2), qeye(2), qeye(2), qeye(2)))
    term23 = (J23/4)*(tensor(qeye(2), sigmax(), sigmax(), qeye(2)) + tensor(qeye(2), sigmay(), sigmay(), qeye(2)) + tensor(qeye(2), sigmaz(), sigmaz(), qeye(2)) - tensor(qeye(2), qeye(2), qeye(2), qeye(2)))
    term34 = (J34/4)*(tensor(qeye(2), qeye(2), sigmax(), sigmax()) + tensor(qeye(2), qeye(2), sigmay(), sigmay()) + tensor(qeye(2), qeye(2), sigmaz(), sigmaz()) - tensor(qeye(2), qeye(2), qeye(2), qeye(2)))
    mfieldterm = B*(tensor(sigmaz(), qeye(2), qeye(2), qeye(2)) + tensor(qeye(2), sigmaz(), qeye(2), qeye(2)) + tensor(qeye(2), qeye(2), sigmaz(), qeye(2)) + tensor(qeye(2), qeye(2), qeye(2), sigmaz()))
    
    term13 = (J13/4)*(tensor(sigmax(), qeye(2), sigmax(), qeye(2)) + tensor(sigmay(), qeye(2), sigmay(), qeye(2)) + tensor(sigmaz(), qeye(2), sigmaz(), qeye(2)) - tensor(qeye(2), qeye(2), qeye(2), qeye(2)))
    term24 = (J24/4)*(tensor(qeye(2), sigmax(), qeye(2), sigmax()) + tensor(qeye(2), sigmay(), qeye(2), sigmay()) + tensor(qeye(2), sigmaz(), qeye(2), sigmaz()) - tensor(qeye(2), qeye(2), qeye(2), qeye(2)))
    
    return term12 + term23 + term34 + mfieldterm + term13 + term24

def hheis_doublet_minus_3_bc(J12, J23, J31, Btilda):
    
    """It give the Heisenberg Hamiltonian (for 3 spins, with bounday conditions) matrix for the doublet subspace (minus) according
    to the analytical formula."""
    
    hheis = np.array([[-J12-(1/4)*(J23+J31)-(Btilda/2),(3/(4*np.sqrt(3)))*(J23-J31)],[(3/(4*np.sqrt(3)))*(J23-J31),(-3/4)*(J23+J31)-(Btilda/2)]])
    
    return hheis

def is_hermitian(H): # Not used, using QuTip version
    
    """Check if an input matrix H (Quantum Object) is Hermitian H = ((H)*)t"""
    
    h_o = np.array(H)
    h_o_ct = np.transpose(np.conjugate(np.array(H)))
    #return np.array_equal(h_o, h_o_ct)
    return np.allclose(h_o, h_o_ct)

def is_unitary(M):
    
    """Check if an input matrix M (Quantum Object) is Unitary U((U)*)t = I"""
    
    m = np.array(M)
    m_ct = np.transpose(np.conjugate(np.array(M)))
    #return np.array_equal(m_ct, np.linalg.inv(m))
    return np.allclose(m_ct, np.linalg.inv(m))