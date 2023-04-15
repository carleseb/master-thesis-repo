#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 09:59:35 2022

@author: ceboncompte
"""

from qutip import *
import numpy as np
import matplotlib.pyplot as plt

def matrix_plot(matrix):
    
    """Plots input matrix (quantum object form). If the matrix is complex it turns it into float."""
    
    plt.figure()
    plt.imshow(np.array(matrix).astype('float64'), alpha=0.8, cmap='YlGn_r')
    plt.xlabel('Column')
    plt.ylabel('Row')
    #plt.title('Matrix')
    plt.colorbar()
    
    return print("SEE PLOT MATRIX")

def basis_transformation(hamiltonian, basis_matrix):
    
    """Returns the original matrix (hamiltonian, Quantum Object) basis-transformed and also as a Quantum Object, given the basis transformation matrix (numpy array)."""
    
    TM = Qobj(basis_matrix)
    H_trans = hamiltonian.transform(TM) # Aparently vectors have to be in rows
    return H_trans

def normal_plot(x_values, y_values):
    
    """It creates a x vs y plot for two given lists."""
    
    plt.figure()
    plt.plot(x_values, y_values)
    plt.xlabel('xaxis')
    plt.ylabel('yaxis')
    plt.title('Title')
    
    return print("SEE PLOT")
    
def Jij(eij, tij):
    
    """Jij formula function."""
    
    return 0.5*(eij + np.sqrt(8*(tij**2) + eij**2))

def energy_diff_doublet_minus_3_bc(J12, J23, J31, Btilda):
    
    """Gives the energy splitting (for 3 spins, with bounday conditions) for the two states in the doublet subspace (minus) according
    to the analytical formula."""
    
    return np.sqrt(J12**2 + J23**2 + J31**2 - J12*J23 - J12*J31 - J23*J31)