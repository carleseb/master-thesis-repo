#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 18:08:32 2022

@author: ceboncompte
"""


from DM_solver.solver  import H_channel, H_solver

from hamiltonian import hheis_general, heisenberg_hamiltonian_3, is_unitary
from use import matrix_plot, basis_transformation
from basis_matrix import coupled_matrix_gen

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


# We first want to create the arbitrary Hamiltonian and print the matrix
Jij_vector = np.array([1]) # in natural units
B = 0.5 # this B is energy! (and its actually Btilda)
spins = len(Jij_vector) + 1

# We check the Hamiltonian is Hermitian
H = hheis_general(Jij_vector, spins, B)
H.check_herm()

# We plot the matrix
matrix_plot(H)

# We generate the basis-transformation matrix
trans_matrix = coupled_matrix_gen(spins)
matrix_plot(trans_matrix) # The basis states are rows
print(trans_matrix[1,:]) #first basis state in the first doublet subspace

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

# We diagonalize the whole subspace
e = sp.linalg.eigvalsh(H_coup)
matrix_plot(np.diag(e)) # it was already diagonal!

# --------------------------------------------- FOR THE ENERGY PLOTS
number_iterations = 100 # number of different values of J we test
energy_tracker = np.zeros((number_iterations, 4)) # 4 (for all states in 2 spins hamiltonian) refers to the number of states in the subspace
Jini = 1.5
Jfin = 0.5
values_J = np.linspace(Jini, Jfin, number_iterations)
#values_J = np.linspace(2, 0, number_iterations)
n = 0

for J in values_J:
    # We first want to create the arbitrary Hamiltonian and print the matrix
    Jij_vector = np.array([J])
    spins = len(Jij_vector) + 1
    
    H = hheis_general(Jij_vector, spins, B)
    
    # We plot the matrix
#    matrix_plot(H)

    # We diagonalize and obtain energy values
    # SInce for two spins in the total spin-spin projection basis the hamiltonian is diagonal, this does nothing in this case
    ev = sp.linalg.eigvalsh(H)
    energy_tracker[n,:] = ev
    n+=1

# we plot the energy of the eigenstates
#plt.figure(figsize=(6,5))
plt.figure()
plt.plot(np.linspace(Jini, Jfin, number_iterations), np.transpose(energy_tracker)[0,:])
plt.plot(np.linspace(Jini, Jfin, number_iterations), np.transpose(energy_tracker)[1,:])
plt.plot(np.linspace(Jini, Jfin, number_iterations), np.transpose(energy_tracker)[2,:])
plt.plot(np.linspace(Jini, Jfin, number_iterations), np.transpose(energy_tracker)[3,:])
plt.xlabel('$J$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
plt.title('Energy of the eigenstates')
plt.show()

"""
# finally we colour plot the expect_tracker matrix data array and the amplitude_tracker for the fourier transform
x = t
y = np.linspace(Jini, Jfin, number_iterations)
plt.pcolormesh(x, y, expect_tracker)
plt.title('Oscillations for the doublet subspace')
plt.xlabel('time ($t_0$)')
plt.ylabel('$J$ ($E_0$)')
plt.colorbar()
plt.show()

# to plot the whole frequency spectrum (with negative ones)
x = sample_freq
y = np.linspace(Jini, Jfin, number_iterations)
plt.pcolormesh(x[:80], y, amplitude_tracker[:,:80])
plt.title('Fourier transfom of the oscillations')
plt.xlabel('frequency ($f_0$)')
plt.ylabel('$J$ ($E_0$)')
plt.colorbar()
plt.show()
"""

# We want to know how the energy levels change varying the magnetic field now
J = 1
Jij_vector = np.array([J]) # in natural units
spins = len(Jij_vector) + 1

# We check the Hamiltonian is Hermitian
H = hheis_general(Jij_vector, spins, B)

number_iterations = 100 # number of different values of J we test
energy_tracker = np.zeros((number_iterations, 4)) # 4 (for all states in 2 spins hamiltonian) refers to the number of states in the subspace
Bini = 0
Bfin = 2
values_B = np.linspace(Bini, Bfin, number_iterations)
#values_B = np.linspace(2, 0, number_iterations)
n = 0

for B in values_B:
    # We first want to create the arbitrary Hamiltonian and print the matrix
    Jij_vector = np.array([J])
    spins = len(Jij_vector) + 1
    
    H = hheis_general(Jij_vector, spins, B)
    
    # We plot the matrix
#    matrix_plot(H)
    
    # We diagonalize and obtain energy values
    # SInce for two spins in the total spin-spin projection basis the hamiltonian is diagonal, this does nothing in this case
    ev = sp.linalg.eigvalsh(H)
    energy_tracker[n,:] = ev
    n+=1

# we plot the energy of the eigenstates
plt.figure()
plt.plot(np.linspace(Bini, Bfin, number_iterations), np.transpose(energy_tracker)[0,:])
plt.plot(np.linspace(Bini, Bfin, number_iterations), np.transpose(energy_tracker)[1,:])
plt.plot(np.linspace(Bini, Bfin, number_iterations), np.transpose(energy_tracker)[2,:])
plt.plot(np.linspace(Bini, Bfin, number_iterations), np.transpose(energy_tracker)[3,:])
plt.xlabel('$\~B$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
plt.title('Energy of the eigenstates')
plt.show()