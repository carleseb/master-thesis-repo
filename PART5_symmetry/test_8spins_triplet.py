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
energy_tracker1 = np.zeros((number_iterations, 3)) # 3 (triplet) refers to the number of states in the subspace
energy_tracker2 = np.zeros((number_iterations, 3))
energy_tracker3 = np.zeros((number_iterations, 3))
energy_tracker4 = np.zeros((number_iterations, 3))
energy_tracker5 = np.zeros((number_iterations, 3))
energy_tracker6 = np.zeros((number_iterations, 3))
J46ini = 1.5
J46fin = 0.5
values_J46 = np.linspace(J46ini, J46fin, number_iterations)
n = 0
J35zero = 2

for J46 in values_J46:
    print('Obtaining energy plots... %', 100*n/number_iterations)
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
    
    Jij_vector = np.array([J35zero - J46, 0, J35zero - J46, 0, J35zero - J46, 0, J35zero - J46])
    Jij_ladder = np.array([J35zero - J46, J35zero - J46, J46, J46, J35zero - J46, J35zero - J46])
    """
    #Jij_vector = np.array([J35zero - J46, 0, J35zero - J46, 0, J46, 0, J46])
    #Jij_ladder = np.array([J35zero - J46, J35zero - J46, 0.66, 0.66, J46, J46])
    
    
    Jij_vector = np.array([0.6, 0, 0.6, 0, 0.6, 0, 0.6])
    Jij_ladder = np.array([J35zero - J46, J46, J35zero - J46 , J46, J35zero - J46, J46])
    """
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
    
    Jij_vector = np.array([J35zero - J46, 0, J35zero - J46, 0, J35zero - J46, 0, J35zero - J46])
    Jij_ladder = np.array([J46, J46, J46, J46, J46, J46])
    
    Jij_vector = np.array([2*J46, 0, 2*(J35zero - J46), 0, 2*(J35zero - J46), 0, 2*J46])
    Jij_ladder = np.array([1, 1, 1, 1, 1, 1])
    
    Jij_vector = np.array([J35zero - J46, 0, J35zero - J46, 0, J35zero - J46, 0, J35zero - J46])
    Jij_ladder = np.array([J46, J46, J46, J46, J46, J46])
    
    Jij_vector = np.array([0.5, 0, J46, 0, J35zero - J46, 0, 0.5])
    Jij_ladder = np.array([1, 1, 1, 1, 1, 1])
    
    Jij_vector = np.array([0.2, 0, 0, 0, 0, 0, 0])
    Jij_ladder = np.array([J35zero - J46, J46, J35zero - J46 , J46, J35zero - J46, J46])
    """
    #Jij_vector = np.array([J46, 0, 0, 0, 0, 0, 1])
    #Jij_ladder = np.array([1, 1, 1, 1, 1, 1])
    
    spins = len(Jij_vector) + 1
    
    H = hheis_general(Jij_vector, spins, B) + ladder_exchanges(Jij_ladder, spins)
    
    # We plot the matrix
#    matrix_plot(H)

    # We finally basis transform and plot again the Hamiltonian matrix
    H_coup = basis_transformation(H, trans_matrix)
#    matrix_plot(H_coup)

    # We take triplets 1-6 and we get the energies of the states
    H_q1 = H_coup[14:17, 14:17]
    H_q2 = H_coup[17:20, 17:20]
    H_q3 = H_coup[20:23, 20:23]
    H_q4 = H_coup[23:26, 23:26]
    H_q5 = H_coup[26:29, 26:29]
    H_q6 = H_coup[29:32, 29:32]
    
    # We diagonalize and obtain energy values
    ev1 = sp.linalg.eigvalsh(H_q1)
    ev2 = sp.linalg.eigvalsh(H_q2)
    ev3 = sp.linalg.eigvalsh(H_q3)
    ev4 = sp.linalg.eigvalsh(H_q4)
    ev5 = sp.linalg.eigvalsh(H_q5)
    ev6 = sp.linalg.eigvalsh(H_q6)
    
    energy_tracker1[n,:] = ev1
    energy_tracker2[n,:] = ev2
    energy_tracker3[n,:] = ev3
    energy_tracker4[n,:] = ev4
    energy_tracker5[n,:] = ev5
    energy_tracker6[n,:] = ev6
    n+=1

n = 0

# triplet 1
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker1)[0,:], label ='E1')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker1)[1,:], label ='E2')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker1)[2,:], label ='E3')
plt.legend()
plt.xlabel('$J_{35} - J_{46}$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
plt.title('Energy of the triplet')
plt.show()

# energy differences triplet 1
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker1)[1,:] - np.transpose(energy_tracker1)[0,:], label ='E2 - E1')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker1)[2,:] - np.transpose(energy_tracker1)[1,:], label ='E3 - E2')
plt.legend()
plt.xlabel('$J_{35} - J_{46}$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
plt.title('Energy differences of the triplet')
plt.show()

# triplet 2
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker2)[0,:], label ='E1')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker2)[1,:], label ='E2')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker2)[2,:], label ='E3')
plt.legend()
plt.xlabel('$J_{35} - J_{46}$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
plt.title('Energy of the triplet')
plt.show()

# triplet 3
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker3)[0,:], label ='E1')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker3)[1,:], label ='E2')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker3)[2,:], label ='E3')
plt.legend()
plt.xlabel('$J_{35} - J_{46}$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
plt.title('Energy of the triplet')
plt.show()

# energy differences triplet 3
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker3)[1,:] - np.transpose(energy_tracker3)[0,:], label ='E2 - E1')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker3)[2,:] - np.transpose(energy_tracker3)[1,:], label ='E3 - E2')
plt.legend()
plt.xlabel('$J_{35} - J_{46}$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
plt.title('Energy differences of the triplet')
plt.show()

# triplet 4
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker4)[0,:], label ='E1')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker4)[1,:], label ='E2')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker4)[2,:], label ='E3')
plt.legend()
plt.xlabel('$J_{35} - J_{46}$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
plt.title('Energy of the triplet')
plt.show()

# triplet 5
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker5)[0,:], label ='E1')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker5)[1,:], label ='E2')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker5)[2,:], label ='E3')
plt.legend()
plt.xlabel('$J_{35} - J_{46}$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
plt.title('Energy of the triplet')
plt.show()

# triplet 6
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker6)[0,:], label ='E1')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker6)[1,:], label ='E2')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker6)[2,:], label ='E3')
plt.legend()
plt.xlabel('$J_{35} - J_{46}$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
plt.title('Energy of the triplet')
plt.show()





# Now we just want to plot triplet 2 but for different values of the fixed exchanges

# --------------------------------------------- FOR THE ENERGY PLOTS
number_iterations = 20 # number of different values of J we test
iter_diff = 11
energy_tracker1 = np.zeros((number_iterations, iter_diff, 3))

J46ini = 1.5
J46fin = 0.5
values_J46 = np.linspace(J46ini, J46fin, number_iterations)
n = 0
J35zero = 2

diffini = 0.2
difffin = 2.2
list_diff = np.linspace(diffini, difffin, iter_diff)

for J46 in values_J46:
    print('Obtaining energy plots... %', 100*n/number_iterations)
    k = 0
    for values_diff in list_diff:
        # We first want to create the arbitrary Hamiltonian and print the matrix
        """
        Jij_vector = np.array([J35zero - J46, 0, J35zero - J46, 0, J35zero - J46, 0, J35zero - J46])
        Jij_ladder = np.array([J46, J46, values_diff, values_diff, J46, J46])
        """
        Jij_vector = np.array([0.2, 0, 0.2, 0, 0.2, 0, values_diff])
        Jij_ladder = np.array([J35zero - J46, J46, 0.2, 0.2, 0.2, 0.2])
        spins = len(Jij_vector) + 1
        
        H = hheis_general(Jij_vector, spins, B) + ladder_exchanges(Jij_ladder, spins)
        
        # We plot the matrix
    #    matrix_plot(H)
    
        # We finally basis transform and plot again the Hamiltonian matrix
        # trans matrix already generated
        H_coup = basis_transformation(H, trans_matrix)
    #    matrix_plot(H_coup)
        
        """
        # We take the sixth triplet and we get the energies of the states
        H_q1 = H_coup[14:17, 14:17]
        H_q2 = H_coup[17:20, 17:20]
        H_q3 = H_coup[20:23, 20:23]
        H_q4 = H_coup[23:26, 23:26]
        H_q5 = H_coup[26:29, 26:29]
        H_q6 = H_coup[29:32, 29:32]   
        
        # We diagonalize and obtain energy values
        ev1 = sp.linalg.eigvalsh(H_q1)
        ev2 = sp.linalg.eigvalsh(H_q2)
        ev3 = sp.linalg.eigvalsh(H_q3)
        ev4 = sp.linalg.eigvalsh(H_q4)
        ev5 = sp.linalg.eigvalsh(H_q5)
        ev6 = sp.linalg.eigvalsh(H_q6)
        
        energy_tracker1[n,k,:] = ev1
        energy_tracker2[n,k,:] = ev2
        energy_tracker3[n,k,:] = ev3
        energy_tracker4[n,k,:] = ev4
        energy_tracker5[n,k,:] = ev5
        energy_tracker6[n,k,:] = ev6
        """
        H_q1 = H_coup[14:17, 14:17]
        ev1 = sp.linalg.eigvalsh(H_q1)
        energy_tracker1[n,k,:] = ev1
        
        k+=1
    n+=1

for i in range(iter_diff):
    plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), energy_tracker1[:,i,0], label ='E1')
    plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), energy_tracker1[:,i,1], label ='E2')
    plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), energy_tracker1[:,i,2], label ='E3')
    plt.legend()
    plt.xlabel('$J_{35} - J_{46}$ ($E_0$)')
    plt.ylabel('energy ($E_0$)')
    plt.title('Energy of the triplet')
    plt.show()