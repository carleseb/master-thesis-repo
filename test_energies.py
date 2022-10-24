#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 14:50:51 2022

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

number_iterations = 100
energy_tracker = np.zeros((number_iterations, 3))
n = 0

for J34 in np.linspace(1.5, 0.5, number_iterations):
    # We first want to create the arbitrary Hamiltonian and print the matrix
    Jij_vector = np.array([2-J34, 0.2, J34]) # in microeV HOW DO WE HAPPEN TO CHOOSE THIS '2' VALUE?
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
    
    # We take the first triplet and we get the energies of the states
    H_triplet = H_coup[2:5, 2:5]
    
    # We diagonalize and obtain energy values
    ev = sp.linalg.eigvalsh(H_triplet)
    energy_tracker[n,:] = ev
    n+=1

plt.figure()
plt.plot(2-2*np.linspace(1.5, 0.5, number_iterations), np.transpose(energy_tracker)[0,:], label ='first triplet list (label 2)')
plt.plot(2-2*np.linspace(1.5, 0.5, number_iterations), np.transpose(energy_tracker)[1,:], label ='second triplet list (label 3)')
plt.plot(2-2*np.linspace(1.5, 0.5, number_iterations), np.transpose(energy_tracker)[2,:], label ='third triplet list (label 4)')
plt.legend()
plt.xlabel('J12 - J34')
plt.ylabel('energy (micro eV)')
plt.title('Energy of the triplets')

# Now we take the value for which we observe minimum energy gap between bot triplets (and homogeneous exchange)
# and we fix the J12 and J34 values, varying the J23 to obtain same value for all exchanges, we will see condition in energy

energy_tracker_ = np.zeros((number_iterations, 3))
n = 0

for J23 in np.linspace(0, 2, number_iterations):
    # We first want to create the arbitrary Hamiltonian and print the matrix
    Jij_vector = np.array([2-1, J23, 1]) # in microeV HOW DO WE HAPPEN TO CHOOSE THIS '2' VALUE?
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
    
    # We take the first triplet and we get the energies of the states
    H_triplet = H_coup[2:5, 2:5]
    
    # We diagonalize and obtain energy values
    ev = sp.linalg.eigvalsh(H_triplet)
    energy_tracker_[n,:] = ev
    n+=1

plt.figure()
plt.plot(np.linspace(0, 2, number_iterations), np.transpose(energy_tracker_)[0,:], label ='first triplet list (label 2)')
plt.plot(np.linspace(0, 2, number_iterations), np.transpose(energy_tracker_)[1,:], label ='second triplet list (label 3)')
plt.plot(np.linspace(0, 2, number_iterations), np.transpose(energy_tracker_)[2,:], label ='third triplet list (label 4)')
plt.legend()
plt.xlabel('J23')
plt.ylabel('energy (micro eV)')
plt.title('Energy of the triplets')