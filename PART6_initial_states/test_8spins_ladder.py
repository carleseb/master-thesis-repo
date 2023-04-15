#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 08:12:57 2023

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
Jij_vector = np.array([2, 0, 2, 0, 2, 0, 2])
Jij_ladder = np.array([1, 1, 1, 1, 1, 1])

#Jij_vector = np.array([0.6, 0, 0.6, 0, 1, 0, 1])
#Jij_ladder = np.array([1, 1, 0.01, 0.01, 1, 1])
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
"""
matrix_plot(H_coup[198:212, 198:212] != 0)
matrix_plot(H_coup[0:14, 0:14] != 0) # here we can see the first (singlet) block
matrix_plot(H_coup[14:42, 14:42] != 0) # here we cas see the second 28 dimensionsal block (triplet)
matrix_plot(H_coup[98:138, 98:138] != 0) # block starts at 98 and ends at 118
matrix_plot(H_coup[198:240, 198:240] != 0)
"""

# We check again if the Hamiltonian is Hermitian
H_coup.check_herm()

# We get the first septuplet and obtain the eigenvectors
H_s = Qobj(H_coup[198:205, 198:205])
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

ket0 = tensor(S13T24, Tminus, Tminus)
ket1 = tensor(Tminus, S13T24, Tminus)
ket2 = tensor(Tminus, Tminus, S13T24)
ket3 = tensor(S, Tminus, Tminus, Tminus)
ket4 = tensor(Tminus, S, Tminus, Tminus)
ket5 = tensor(Tminus, Tminus, S, Tminus)
ket6 = tensor(Tminus, Tminus, Tminus, S)
ket7 = tensor(T13S24, Tminus, Tminus)
ket8 = tensor(Tminus, T13S24, Tminus)
ket9 = tensor(Tminus, Tminus, T13S24)

ket0_t = basis_transformation(ket0, trans_matrix)
ket0_t
ket0_red = (basis(7,2) + np.sqrt(3)*basis(7,6)).unit()
ket0_red

ket1_t = basis_transformation(ket1, trans_matrix)
ket1_t
ket1_red = (basis(7,0) + np.sqrt(15)*basis(7,1) - np.sqrt(8)*basis(7,6)).unit()
ket1_red

ket2_t = basis_transformation(ket2, trans_matrix)
ket2_t
ket2_red = (-np.sqrt(24)*basis(7,1) + basis(7,4) + np.sqrt(35)*basis(7,5)).unit()
ket2_red

ket3_t = basis_transformation(ket3, trans_matrix)
ket3_t
ket3_red = basis(7,2)
ket3_red

ket4_t = basis_transformation(ket4, trans_matrix)
ket4_t
ket4_red = (np.sqrt(2)*basis(7,0) - basis(7,6)).unit()
ket4_red

ket5_t = basis_transformation(ket5, trans_matrix)
ket5_t
ket5_red = (-np.sqrt(2)*basis(7,1) + np.sqrt(3)*basis(7,4)).unit()
ket5_red

ket6_t = basis_transformation(ket6, trans_matrix)
ket6_t
ket6_red = (np.sqrt(20)*basis(7,3) - np.sqrt(15)*basis(7,5)).unit()
ket6_red

ket7_t = basis_transformation(ket7, trans_matrix)
ket7_t
ket7_red = (np.sqrt(8)*basis(7,0) - np.sqrt(3)*basis(7,2) + basis(7,6)).unit()
ket7_red

ket8_t = basis_transformation(ket8, trans_matrix)
ket8_t
ket8_red = (-np.sqrt(15)*basis(7,0) + basis(7,1) + np.sqrt(24)*basis(7,4)).unit()
ket8_red

ket9_t = basis_transformation(ket9, trans_matrix)
ket9_t
ket9_red = (np.sqrt(48)*basis(7,3) - np.sqrt(35)*basis(7,4) + basis(7,5)).unit()
ket9_red

"""
for index, element in enumerate(ket9_t):
    if element != 0.:
        print(index, element)
"""

ini_states = np.array([ket0_red, ket1_red, ket2_red, ket3_red, ket4_red, ket5_red, ket6_red, ket7_red, ket8_red, ket9_red])
ini_states
len(ini_states)

# Finally we compute the overlaps
overlaps = np.zeros((len(eigenvecs), len(ini_states)))
for i, evec in enumerate(eigenvecs):
    for j, inis in enumerate(ini_states):
        overlaps[i,j] = np.abs(evec.overlap(Qobj(inis)))
        
overlaps.round(2)

# Overlaps matrix stores the overlap of the first eigenvector with all the initial states in the first index
# Therefore we have the eigenvectors on the index list labels and the initial states on the columns labels
overlaps[0]

# We plot the table
fig, ax = plt.subplots()
# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
df = pd.DataFrame(overlaps, index =['e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7'], columns=['i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9', 'i10'])
ax.table(cellText=df.values, rowLabels = df.index, colLabels=df.columns, loc='center')
fig.tight_layout()
plt.show()