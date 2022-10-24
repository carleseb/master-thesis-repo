#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 10:01:19 2022

@author: ceboncompte
"""

from DM_solver.solver  import H_channel, H_solver

from hamiltonian import hheis_general
from use import matrix_plot, basis_transformation
from basis_matrix import coupled_matrix_gen

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

number_iterations = 50
energy_tracker = np.zeros((number_iterations, 2))
n = 0
# We investigate first the 3-dot array

for J23 in np.linspace(3, -1, number_iterations):
    # We first want to create the arbitrary Hamiltonian and print the matrix
    Jij_vector = np.array([2-J23, J23]) # in microeV
    B = 0
    spins = len(Jij_vector) + 1
    
    H = hheis_general(Jij_vector, spins, B)
    
    # We plot the matrix
    matrix_plot(H)
    
    # We generate the basis-transformation matrix
    trans_matrix = coupled_matrix_gen(spins)
    
    # We finally basis transform and plot again the Hamiltonian matrix
    H_coup = basis_transformation(H, trans_matrix)
    matrix_plot(H_coup)
    
    # We take the first triplet and we get the energies of the states NOT TRIPLET HERE
    H_triplet = H_coup[2:4, 2:4]
    
    # We diagonalize and obtain energy values
    ev = sp.linalg.eigvalsh(H_triplet)
    energy_tracker[n,:] = ev
    n+=1

plt.figure()
plt.plot(2-2*np.linspace(3, -1, number_iterations), np.transpose(energy_tracker)[0,:])
plt.plot(2-2*np.linspace(3, -1, number_iterations), np.transpose(energy_tracker)[1,:])
plt.xlabel('J12 - J23')
plt.ylabel('energy (micro eV)')
plt.title('Energy of the doublets')


# We can watch the 5- dot array
energy_tracker_ = np.zeros((number_iterations, 4))
n=0

for J45 in np.linspace(3, -1, number_iterations):
    Jij_vector = np.array([2-J45, 0.1, 0.1, J45]) # in microeV
    B = 0
    spins = len(Jij_vector) + 1
    
    H = hheis_general(Jij_vector, spins, B)
    
    # We plot the matrix
    matrix_plot(H)
    
    # We generate the basis-transformation matrix
    trans_matrix = coupled_matrix_gen(spins)
    
    # We finally basis transform and plot again the Hamiltonian matrix
    H_coup = basis_transformation(H, trans_matrix)
    matrix_plot(H_coup)
    
    # We take the fist quadruplets
    H_quadruplet = H_coup[10:14,10:14]
    
    # We diagonalize and obtain energy values
    ev = sp.linalg.eigvalsh(H_quadruplet)
    energy_tracker_[n,:] = ev
    n+=1

plt.figure()
plt.plot(2-2*np.linspace(3, -1, number_iterations), np.transpose(energy_tracker_)[0,:], label='first quadruplet list (label 10)')
plt.plot(2-2*np.linspace(3, -1, number_iterations), np.transpose(energy_tracker_)[1,:], label='second quadruplet list (label 11)')
plt.plot(2-2*np.linspace(3, -1, number_iterations), np.transpose(energy_tracker_)[2,:], label='third quadruplet list (label 12)')
plt.plot(2-2*np.linspace(3, -1, number_iterations), np.transpose(energy_tracker_)[3,:], label='fourth quadruplet list (label 13)')
plt.legend()
plt.xlabel('J12 - J45')
plt.ylabel('energy (micro eV)')
plt.title('Energy of the quadruplets')

# Now we attempt to do the same as in test_energies but with two Js in between instead of 1

energy_tracker__ = np.zeros((number_iterations, 4))
n=0

for J3 in np.linspace(0, 2, number_iterations):
    Jij_vector = np.array([2-1, J3, J3, 1]) # in microeV
    B = 0
    spins = len(Jij_vector) + 1
    
    H = hheis_general(Jij_vector, spins, B)
    
    # We plot the matrix
    matrix_plot(H)
    
    # We generate the basis-transformation matrix
    trans_matrix = coupled_matrix_gen(spins)
    
    # We finally basis transform and plot again the Hamiltonian matrix
    H_coup = basis_transformation(H, trans_matrix)
    matrix_plot(H_coup)
    
    # We take the fist quadruplets
    H_quadruplet = H_coup[10:14,10:14]
    
    # We diagonalize and obtain energy values
    ev = sp.linalg.eigvalsh(H_quadruplet)
    print(ev, J3)
    energy_tracker__[n,:] = ev
    n+=1

plt.figure()
plt.plot(np.linspace(0, 2, number_iterations), np.transpose(energy_tracker__)[0,:], label='first quadruplet list (label 10)')
plt.plot(np.linspace(0, 2, number_iterations), np.transpose(energy_tracker__)[1,:], label='second quadruplet list (label 11)')
plt.plot(np.linspace(0, 2, number_iterations), np.transpose(energy_tracker__)[2,:], label='third quadruplet list (label 12)')
plt.plot(np.linspace(0, 2, number_iterations), np.transpose(energy_tracker__)[3,:], label='fourth quadruplet list (label 13)')
plt.legend()
plt.xlabel('J23 = J34')
plt.ylabel('energy (micro eV)')
plt.title('Energy of the quadruplets')