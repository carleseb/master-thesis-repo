#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 16:28:52 2023

@author: ceboncompte
"""

from hamiltonian import hheis_general, ladder_exchanges, chain_bc, heisenberg_hamiltonian_4, is_unitary
from use import matrix_plot, basis_transformation
from basis_matrix import coupled_matrix_gen

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# Number of spins for this simulation and magnetic field
spins = 4
B = 0.5

# Two spins basis states
S = (basis(4,1) - basis(4,2)).unit()
T0 = (basis(4,1) + basis(4,2)).unit()
Tplus = basis(4,0)
Tminus = basis(4,3)

# We build the general pair-product states
TM0 = tensor(S,Tminus)
TM1 = tensor(Tminus,S)
TM2 = (tensor(T0,Tminus) - tensor(Tminus,T0)).unit()
print(TM0)

# We generate the basis-transformation matrix and we transform the states
trans_matrix = coupled_matrix_gen(spins)
matrix_plot(trans_matrix)
TM0_t = basis_transformation(TM0, trans_matrix)
TM1_t = basis_transformation(TM1, trans_matrix).unit()
TM2_t = basis_transformation(TM2, trans_matrix).unit()
k1 = basis_transformation(TM1, trans_matrix)
k2 = basis_transformation(TM2, trans_matrix)
sup = (TM1_t + TM2_t).unit()
print(np.array(TM0_t))
print(np.array(TM1_t), k1)
print(np.array(TM2_t), k2) #ok

TTT = (tensor(T0, Tminus) - tensor(Tminus, T0)).unit()
print(TTT)

TTT_t = basis_transformation(TTT, trans_matrix)
print(TTT_t)

B = 0
Jij_vector = np.array([1, 1, 1])
spins = len(Jij_vector) + 1
H = hheis_general(Jij_vector, spins, B)
matrix_plot(H)

trans_matrix = coupled_matrix_gen(spins)
H_coup = basis_transformation(H, trans_matrix)
matrix_plot(H_coup)

H_triplet = H_coup[2:5, 2:5]
matrix_plot(H_triplet)
print(H_triplet)

# We plot
fig, ax = plt.subplots()
ax.set_xlabel('Column')
ax.set_ylabel('Row')
im = ax.imshow(np.real(H_triplet))
fig.colorbar(im)
intersection_matrix = np.round_(np.real(H_triplet), decimals = 2)
ax.matshow(intersection_matrix)
for i in range(3):
    for j in range(3):
        c = intersection_matrix[j,i]
        ax.text(i, j, str(c), va='center', ha='center')

# Let's see the differen representation so of pair states

s1 = tensor(S, S)
print(s1)

# What shape does this vector have in our total spin basis
s1_t = basis_transformation(s1, trans_matrix)
print(s1_t)

# Let's see the second state of the singlet
comput2 = tensor(basis(2,0), basis(2,0), basis(2,0), basis(2,1))
print(comput2)

comput2_t = basis_transformation(comput2, Qobj(trans_matrix).dag())
print(comput2_t)

s2 = (tensor(Tplus, Tminus) + tensor(Tminus, Tplus) - 2*tensor(T0, T0)).unit()
s2p = (tensor(Tplus, Tminus) + tensor(Tminus, Tplus) - 2*tensor(T0, T0))*(1/np.sqrt(3))
print(s2, s2p)

ss2 = (tensor(Tplus, Tminus) + tensor(Tminus, Tplus) - tensor(T0, T0)).unit()
print(ss2)

# Let's get the all connected four spins Hamiltonian and examine the singlet subspace
Jij_vector = np.array([1, 1, 1])
Jij_ladder = np.array([1, 1])
Jbc = 1
B = 0

spins = len(Jij_vector) + 1
H = hheis_general(Jij_vector, spins, B) + ladder_exchanges(Jij_ladder, spins) + chain_bc(Jbc, spins)
matrix_plot(H)

trans_matrix = coupled_matrix_gen(spins)
H_coup = basis_transformation(H, trans_matrix)
matrix_plot(H_coup)

H_singlet = H_coup[0:2, 0:2]
matrix_plot(H_singlet)
print(H_singlet)