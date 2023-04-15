#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 16:22:26 2022

@author: ceboncompte
"""

from DM_solver.solver  import H_channel, H_solver

from hamiltonian import hheis_general, ladder_exchanges, is_unitary
from use import matrix_plot, basis_transformation
from basis_matrix import coupled_matrix_gen

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# We first want to create the arbitrary Hamiltonian and print the matrix
Jij_vector = np.array([1, 0, 1, 0, 1, 0, 1])
Jij_ladder = np.array([1, 1, 1, 1, 1, 1])
B = 0.5
spins = len(Jij_vector) + 1

# We check the Hamiltonian is Hermitian
H = hheis_general(Jij_vector, spins, B) + ladder_exchanges(Jij_ladder, spins)
H.check_herm()

# We plot the matrix
matrix_plot(H)

# We generate the basis-transformation matrix
trans_matrix = coupled_matrix_gen(spins)
matrix_plot(trans_matrix) # The basis states are rows
print(trans_matrix[0,:]) # first basis state

# We also want to check if the basis-transformation matrix is unitary
ct_trans_matrix = np.transpose(np.conjugate(trans_matrix))
matrix_plot(np.matmul(trans_matrix, ct_trans_matrix))
# We see we get an identity
# Alternatively (unitary test works with allclose numpy function)
is_unitary(trans_matrix)

# We finally basis transform and plot again the Hamiltonian matrix
H_coup = basis_transformation(H, trans_matrix)
# And we plot it
matrix_plot(H_coup)

# We check again if the Hamiltonian is Hermitian
H_coup.check_herm()

# --------------------------------------------- FOR THE ENERGY PLOTS
number_iterations = 20 # number of different values of J we test
energy_tracker = np.zeros((number_iterations, 5)) # 5 (quintuplet) refers to the number of states in the subspace
J46ini = 1.5
J46fin = 0.5
values_J46 = np.linspace(J46ini, J46fin, number_iterations)
n = 0
J35zero = 2

for J46 in values_J46:
    print('Obtaining energy plots... %', n*100/number_iterations)
    # We first want to create the arbitrary Hamiltonian and print the matrix
    """
    Jij_vector = np.array([J35zero - J46, 0, J35zero - J46, 0, J46, 0, J46])
    Jij_ladder = np.array([J35zero - J46, J35zero - J46, 0.8, 0.55, J46, J46])
    
    Jij_vector = np.array([0.6, 0, 0.6, 0, 0.6, 0, 0.6])
    Jij_ladder = np.array([J35zero - J46, J46, J35zero - J46 , J46, J35zero - J46, J46])
    
    Jij_vector = np.array([0.6, 0, J35zero - J46, 0, J46, 0, 0.6])
    Jij_ladder = np.array([1, 1, 1, 1, 1, 1])
    
    Jij_vector = np.array([1, 0, 1, 0, 1, 0, 1])
    Jij_ladder = np.array([1, J35zero - J46, 1, 1, J46, 1])
    
    Jij_vector = np.array([1, 0, 1, 0, 1, 0, 1])
    Jij_ladder = np.array([J35zero - J46, J46, 1, 1, 1, 1])
    
    Jij_vector = np.array([1, 0, 1, 0, 1, 0, 1])
    Jij_ladder = np.array([J35zero - J46, 1, 1, 1, J46, 1])
    
    Jij_vector = np.array([1, 0, 1, 0, 1, 0, 1])
    Jij_ladder = np.array([1, 1, 2*(J35zero - J46), 2*J46, 1, 1])
    
    Jij_vector = np.array([1, 0, 1, 0, 1, 0, 1])
    Jij_ladder = np.array([1, 1, J46, 1, 1, 1])
    
    Jij_vector = np.array([1, 0, 1, 0, 1, 0, 1])
    Jij_ladder = np.array([1, 1, 2*(J35zero - J46), 2*J46, 1, 1])
    
    Jij_vector = np.array([J35zero - J46, 0, J35zero - J46, 0, J35zero - J46, 0, J35zero - J46])
    Jij_ladder = np.array([J35zero - J46, J35zero - J46, J46, J46, J35zero - J46, J35zero - J46])
    """
    #Jij_vector = np.array([J35zero - J46, 0, J35zero - J46, 0, J35zero - J46, 0, J35zero - J46])
    #Jij_ladder = np.array([J35zero - J46, J35zero - J46, J46, J35zero - J46, J35zero - J46, J35zero - J46])
    
    """
    Jij_vector = np.array([0.6, 0, 0.6, 0, 0.6, 0, 0.6])
    Jij_ladder = np.array([J35zero - J46, J46, J35zero - J46 , J46, J35zero - J46, J46])
    
    Jij_vector = np.array([0.4, 0, 0.3, 0, 0.8, 0, 0.75])
    Jij_ladder = np.array([J35zero - J46, J46, J35zero - J46 , J46, J35zero - J46, J46])
    
    Jij_vector = np.array([0, 0, J46, 0, J35zero - J46, 0, 0])
    Jij_ladder = np.array([1, 1, 1, 1, 1, 1])
    
    Jij_vector = np.array([0.5, 0, J46, 0, J35zero - J46, 0, 0.5])
    Jij_ladder = np.array([1, 1, 1, 1, 1, 1])
    
    Jij_vector = np.array([2*J46, 0, 2*(J35zero - J46), 0, 2*(J35zero - J46), 0, 2*J46])
    Jij_ladder = np.array([1, 1, 1, 1, 1, 1])
    
    Jij_vector = np.array([0.5*J46, 0, 0.5*(J35zero - J46), 0, 0.5*(J35zero - J46), 0, 0.5*J46])
    Jij_ladder = np.array([1, 1, 1, 1, 1, 1])
    """
    #Jij_vector = np.array([J35zero - J46, 0, J35zero - J46, 0, J35zero - J46, 0, J35zero - J46])
    #Jij_ladder = np.array([J46, J46, J46, J46, J46, J46])
    
    Jij_vector = np.array([J46, 0, 0, 0, 0, 0, 1])
    Jij_ladder = np.array([1, 1, 1, 1, 1, 1])
    
    spins = len(Jij_vector) + 1
    
    H = hheis_general(Jij_vector, spins, B) + ladder_exchanges(Jij_ladder, spins)
    
    # We plot the matrix
#    matrix_plot(H)
    
    # We finally basis transform and plot again the Hamiltonian matrix
    H_coup = basis_transformation(H, trans_matrix)
#    matrix_plot(H_coup)
    
    # We take the first quintuplet and we get the energies of the states
    H_q = H_coup[98:103, 98:103]
    #H_q = H_coup[103:108, 103:108]
    #H_q = H_coup[108:113, 108:113]
    
    # We diagonalize and obtain energy values
    ev = sp.linalg.eigvalsh(H_q)
    energy_tracker[n,:] = ev
    n+=1

n = 0

# we plot the energy of the eigenstates
plt.figure()
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[0,:], label ='E1')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[1,:], label ='E2')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[2,:], label ='E3')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[3,:], label ='E4')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[4,:], label ='E5')
plt.legend()
plt.xlabel('$J_{35} - J_{46}$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
plt.title('Energies of the quintuplet subspace')
plt.show()

# we plot the differences of succesive eigenstate's energies
plt.figure()
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[1,:] - np.transpose(energy_tracker)[0,:], label ='E2 - E1')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[2,:] - np.transpose(energy_tracker)[1,:], label ='E3 - E2')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[3,:] - np.transpose(energy_tracker)[2,:], label ='E4 - E3')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[4,:] - np.transpose(energy_tracker)[3,:], label ='E5 - E4')
plt.legend()
plt.xlabel('$J_{35} - J_{46}$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
plt.title('Energy differences of adjacent states of the quintuplet subspace')
plt.show()

# we plot the differences of succesive eigenstate's energies
plt.figure()
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[3,:] - np.transpose(energy_tracker)[1,:], label ='E4 - E2')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[4,:] - np.transpose(energy_tracker)[3,:], label ='E5 - E4')
plt.legend()
plt.xlabel('$J_{35} - J_{46}$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
plt.title('Energy differences of adjacent states of the quintuplet subspace')
plt.show()

# we plot the differences of succesive eigenstate's energies
plt.figure()
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[2,:] - np.transpose(energy_tracker)[0,:], label ='E3 - E1')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[3,:] - np.transpose(energy_tracker)[2,:], label ='E4 - E3')
plt.legend()
plt.xlabel('$J_{35} - J_{46}$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
plt.title('Energy differences of adjacent states of the quintuplet subspace')
plt.show()