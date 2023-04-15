#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 08:07:39 2023

@author: ceboncompte
"""

from DM_solver.solver  import H_channel, H_solver

from hamiltonian import hheis_general, heisenberg_hamiltonian_3, ladder_exchanges, is_unitary
from use import matrix_plot, basis_transformation
from basis_matrix import coupled_matrix_gen

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd

# We first want to create the arbitrary Hamiltonian and print the matrix
Jij_vector = np.array([1, 0, 1]) # in natural units
Jij_ladder = np.array([1, 1]) # in natural units
B = 0.5 # this B is energy!
spins = len(Jij_vector) + 1

# We check the Hamiltonian is Hermitian
H = hheis_general(Jij_vector, spins, B) + ladder_exchanges(Jij_ladder, spins)
H.check_herm()

# We plot the matrix
matrix_plot(H)

# We generate the basis-transformation matrix
trans_matrix = coupled_matrix_gen(spins)
matrix_plot(trans_matrix) # The basis states are rows

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

# We get the first triplet
H_s = Qobj(H_coup[2:5, 2:5])
H_s

eigenvals, eigenvecs = H_s.eigenstates()
eigenvecs
eigenvecs[0]

# We define our set of initializable states
# Intitial state is the first state (first row) of the basis-transformation matrix
S = (basis(4,1) - basis(4,2)).unit()
Tminus = basis(4,3)

S13T24 = (basis(16,7) - basis(16,13)).unit()
T13S24 = (basis(16,11) - basis(16,14)).unit()

ket0 = tensor(S13T24)
ket1 = tensor(S, Tminus)
ket2 = tensor(Tminus, S)
ket3 = tensor(T13S24)

ket0_t = basis_transformation(ket0, trans_matrix)
ket0_t
ket0_red = (basis(3,0) + np.sqrt(3)*basis(3,1)).unit()
ket0_red

ket1_t = basis_transformation(ket1, trans_matrix)
ket1_t
ket1_red = basis(3,0)
ket1_red

ket2_t = basis_transformation(ket2, trans_matrix)
ket2_t
ket2_red = (-basis(3,1) + np.sqrt(2)*basis(3,2)).unit()
ket2_red

ket3_t = basis_transformation(ket3, trans_matrix)
ket3_t
ket3_red = (-np.sqrt(3)*basis(3,0) + basis(3,1) + np.sqrt(8)*basis(3,2)).unit()
ket3_red

"""
for index, element in enumerate(ket3_t):
    if element != 0.:
        print(index, element**2)
"""

ini_states = np.array([ket0_red, ket1_red, ket2_red, ket3_red])
ini_states
len(ini_states)

# Finally we compute the overlaps
overlaps = np.zeros((len(eigenvecs), len(ini_states)))
for i, evec in enumerate(eigenvecs):
    for j, inis in enumerate(ini_states):
        overlaps[i,j] = np.abs(evec.overlap(Qobj(inis)))
        
overlaps.round(2)