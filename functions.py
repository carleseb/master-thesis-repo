#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 14:18:57 2022

@author: ceboncompte
"""

from qutip import *
import numpy as np
import matplotlib.pyplot as plt


def heisenberg_hamiltonian(J12, J23, J34, B):

    """Returns a Heisenberg Hamiltonian in matrix form in the computational basis as a quantum object given the input parameters."""
    
    term12 = (J12/4)*(tensor(sigmax(), sigmax(), qeye(2), qeye(2)) + tensor(sigmay(), sigmay(), qeye(2), qeye(2)) + tensor(sigmaz(), sigmaz(), qeye(2), qeye(2)) - tensor(qeye(2), qeye(2), qeye(2), qeye(2)))
    term23 = (J23/4)*(tensor(qeye(2), sigmax(), sigmax(), qeye(2)) + tensor(qeye(2), sigmay(), sigmay(), qeye(2)) + tensor(qeye(2), sigmaz(), sigmaz(), qeye(2)) - tensor(qeye(2), qeye(2), qeye(2), qeye(2)))
    term34 = (J34/4)*(tensor(qeye(2), qeye(2), sigmax(), sigmax()) + tensor(qeye(2), qeye(2), sigmay(), sigmay()) + tensor(qeye(2), qeye(2), sigmaz(), sigmaz()) - tensor(qeye(2), qeye(2), qeye(2), qeye(2)))
    mfieldterm = B*(tensor(sigmaz(), qeye(2), qeye(2), qeye(2)) + tensor(qeye(2), sigmaz(), qeye(2), qeye(2)) + tensor(qeye(2), qeye(2), sigmaz(), qeye(2)) + tensor(qeye(2), qeye(2), qeye(2), sigmaz()))
    
    return term12 + term23 + term34 + mfieldterm

def matrix_plot(matrix):
    
    """Plots input matrix (quantum object form). If the matrix is complex it turns it into float."""
    
    plt.figure()
    plt.imshow(np.array(matrix).astype('float64'), alpha=0.8, cmap='YlGn_r')
    plt.xlabel('Row')
    plt.ylabel('Column')
    plt.title('Matrix')
    plt.colorbar()
    
    return print("SEE PLOT")

def A(j1, j2, j): # Called in CG
    
    """First term of the Clebsh Gordan coefficients."""
    
    T1 = np.math.factorial(j1 + j2 - j)
    T2 = np.math.factorial(j + j1 - j2)
    T3 = np.math.factorial(j2 + j - j1)
    T4 = np.math.factorial(j1 + j2 + j + 1)
    return (T1*T2*T3)/T4

def B(j1, j2, j, m1, m2, m): # Called in CG
    
    """Second term of the Clebsh Gordan coefficients."""
    
    if j1 + m1 >= 0 and j1 - m1 >= 0 and j2 + m2 >= 0 and j2 - m2 >= 0 and j + m >= 0 and j - m >= 0:
        T1 = np.math.factorial(j1 + m1)
        T2 = np.math.factorial(j1 - m1)
        T3 = np.math.factorial(j2 + m2)
        T4 = np.math.factorial(j2 - m2)
        T5 = np.math.factorial(j + m)
        T6 = np.math.factorial(j - m)
        return T1*T2*T3*T4*T5*T6
    else:
        return 0

def C(j1, j2, j, m1, m2, v): # Called in CG
    
    """Third term term of the Clebsh Gordan coefficients."""
    
    T1 = np.math.factorial(j1 + j2 - j - v)
    T2 = np.math.factorial(j1 - m1 - v)
    T3 = np.math.factorial(j2 + m2 - v)
    T4 = np.math.factorial(j - j2 + m1 + v)
    T5 = np.math.factorial(j - j1 - m2 + v)
    return T1*T2*T3*T4*T5

def CG(j1, j2, j, m1, m2, m): # Called in coupled_basis_matrix
    
    """Gives Clebsh Gordan coefficient for two given spins in the computational basis (total spin and z-axis projection needed for both spins)."""
    
    if m == m1+m2:
        T1 = ((2*j + 1)*A(j1, j2, j)*B(j1, j2, j, m1, m2, m))**0.5
        T2 = 0
        for v in range(8): # This needs to be revised
            if j1 + j2 - j - v >= 0 and j1 - m1 - v >= 0 and j2 + m2 - v >= 0 and j - j2 + m1 + v >= 0 and j - j1 - m2 + v >= 0:
                T2 += (((-1)**v)*(C(j1, j2, j, m1, m2, v))**(-1))/np.math.factorial(v)
            else:
                continue
        return T1*T2
    else:
        return 0

def compose(j1, j2): # Called in labelling
    
    """Returns a list with ordered (from lower to higher) possible total spins in the spin composition."""
    
    list12 = np.array([])

    n12 = 0
    while j1+j2-n12 >= 0 and j1+j2-n12 >= j1-j2:
        element12 = j1+j2-n12
        list12 = np.append(list12, element12)
        n12 += 1
    list12 = np.sort(list12)
    return list12

def labelling(spins): # Called in coupled_basis_matrix # This function only works for spin 1/2 and four spins xD
    
    """Returns a list of labels for the coupled basis Hilbert space vectors. The list has '2**spins' rows and 'spins' columns, the number of labels needed for every vector."""
    
    j1 = j2 = j3 = j4 = 1/2
    #spins = 4
    
    labels = np.zeros((2**spins, spins), dtype = float)
    comp12 = compose(j1, j2)
    u = 0

    for totalspin12 in comp12:
        comp23 = compose(totalspin12, j3)
        for totalspin123 in comp23:
            comp34 = compose(totalspin123, j4)
            for totalspin in comp34:
                k = 0
                while totalspin - k >= - totalspin:
                    labels[u,0] = totalspin12
                    labels[u,1] = totalspin123
                    labels[u,2] = totalspin
                    labels[u,3] = totalspin - k
                    k +=1
                    u +=1
    return labels

def ordering(labels): # Called in coupled_basis_matrix # This is under construction, either do it by hand or rebuild
    
    """Orders the labels of the coupled basis Hilbert space to show bloch diagonal Hamiltonians in the cases of interest."""
    
    labels[[1, 4]] = labels[[4, 1]]
    labels[[3, 6]] = labels[[6, 3]]
    labels[[4, 9]] = labels[[9, 4]]
    labels[[6, 8]] = labels[[8, 6]]
    labels[[7, 9]] = labels[[9, 7]]
    
    return print("LABELS ORDERED")

def coupled_basis_matrix(spins): # This function only works for spin 1/2 and four spins xD
    
    """Returns the base-switch matrix from computational basis to coupled basis"""
    
    j1 = j2 = j3 = j4 = 1/2
    #spins = 4
    fourspinmat = np.zeros((2**spins, 2**spins), dtype = complex)
    
    labels = labelling(spins)
    ordering(labels)
    
    for i in range(2**spins): # Vector index
        eigenvector = 0 # We reset the eigenvector
        n = 0 # Basis index
        for m1 in 1/2,-1/2:
            for m2 in 1/2,-1/2:
                for m3 in 1/2,-1/2:
                    for m4 in 1/2,-1/2:
                        m12 = m1 + m2
                        m123 = m12 + m3

                        j12 = labels[i,0] # First index vector index
                        j123 = labels[i,1]
                        j = labels[i,2]
                        m = labels[i,3]

                        eigenvector += (CG(j1, j2, j12, m1, m2, m12)*CG(j12, j3, j123, m12, m3, m123)*CG(j123, j4, j, m123, m4, m))*np.array(basis(2**spins, n).trans())[0]
                        n+=1

        fourspinmat[i,:] = eigenvector # Now we put the eigenvector into the matrix and follow with the next ones
    return fourspinmat

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