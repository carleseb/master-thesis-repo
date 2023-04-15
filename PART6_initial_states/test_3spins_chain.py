#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 10:01:19 2022

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
import pandas as pd

# We first want to create the arbitrary Hamiltonian and print the matrix
Jij_vector = np.array([1, 1]) # in natural units
B = 0.5 # this B is energy!
spins = len(Jij_vector) + 1

# We check the Hamiltonian is Hermitian
H = hheis_general(Jij_vector, spins, B)
H.check_herm()

# We plot the matrix
matrix_plot(H)

# We first want to check if the Hamiltonian that our general function yeilds is the same as we expect
# from a 3 spins Heisenberg Hamiltonian
H3 = heisenberg_hamiltonian_3(1, 1, B)
H == H3
# We see that for any combination of Js and B we get exactly the same matrix

# We generate the basis-transformation matrix
trans_matrix = coupled_matrix_gen(spins)
matrix_plot(trans_matrix) # The basis states are rows
print(trans_matrix[0,:]) #first basis state in the first doublet subspace
print(trans_matrix[1,:]) #second

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

# We see for 3 spins we have 2 doublets (degenerate in energy) and 1 quadruplet
# We get the first doublet and obtain the eigenvectors
H_doublet = Qobj(H_coup[0:2,0:2])
H_doublet

eigenvals, eigenvecs = H_doublet.eigenstates()
eigenvecs
eigenvecs[0]

# We define our set of initializable states
# Intitial state is the first state (first row) of the basis-transformation matrix
S = (basis(4,1) - basis(4,2)).unit()
ket0 = tensor(S, basis(2,1))
print(ket0)
ket0_t = basis_transformation(ket0, trans_matrix)
print(ket0_t) #this is basis(2,0)
ket0_red = basis(2,0)
ket0_red

# The other state composing the RVB
ket1 = tensor(basis(2,1), S)
print(ket1)
ket1_t = basis_transformation(ket1, trans_matrix)
print(ket1_t) #this is -basis(2,0)+sqrt(3)*basis(2,1)
ket1_red = (-basis(2,0) + np.sqrt(3)*basis(2,1)).unit()
print(ket1_red)
ket1_red

ini_states = np.array([ket0_red, ket1_red])
ini_states
len(ini_states)

# Finally we compute the overlaps
overlaps = np.zeros((len(eigenvecs), len(ini_states)))
for i, evec in enumerate(eigenvecs):
    for j, inis in enumerate(ini_states):
        overlaps[i,j] = np.abs(evec.overlap(Qobj(inis)))

# Overlaps matrix stores the overlap of the first eigenvector with all the initial states in the first index
# Therefore we have the eigenvectors on the index list labels and the initial states on the columns labels
overlaps[0]

# We plot the table
fig, ax = plt.subplots()
# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
df = pd.DataFrame(overlaps, index =['eigenvector 1', 'eigenvector 2'], columns=['initial state 1', 'initial state 2'])
ax.table(cellText=df.values, rowLabels = df.index, colLabels=df.columns, loc='center')
fig.tight_layout()
plt.show()
