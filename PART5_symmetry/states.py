#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 15:41:35 2023

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

# --------------------------------------------- LOOK 6 SPINS CHAIN SCRIPT FOR MORE

# For the initial state we examine the pairs basis
S = (basis(4,1) - basis(4,2)).unit()
T0 = (basis(4,1) + basis(4,2)).unit()
Tplus = basis(4,0)
Tminus = basis(4,3)

p1 = (basis(16, 5) - basis(16,6) - basis(16,9) + basis(16,10)).unit() #S12, S34
print(p1)
#p11 = tensor(S, S) #S12, S34
#print(p11)
p2 = (basis(16, 3) - basis(16,5) - basis(16,10) + basis(16,12)).unit() # S14 S23
print(p2) #S14 S23 our notation
p3 = (basis(16, 3) - basis(16,6) - basis(16,9) + basis(16,12)).unit() #S13 S24
print(p3) # our notation
S13S24 = p3

S13T24 = (basis(16,7) - basis(16,13)).unit()
print(S13T24)

T13S24 = (basis(16,11) - basis(16,14)).unit()
print(T13S24)

# --- SEPTUPLETS

# We propose 
state = tensor(S, Tminus, Tminus, Tminus)
state_t = basis_transformation(state, trans_matrix).unit()
print(state_t) # all zeros except one 1 at position 200
# this state belongs to the first setuplet [198:205] (205 not included)

state = tensor(Tminus, S, Tminus, Tminus)
state_t = basis_transformation(state, trans_matrix).unit()
print(state_t) # also lives in the first septuplet

state = tensor(Tminus, Tminus, S, Tminus)
state_t = basis_transformation(state, trans_matrix).unit()
print(state_t) # also lives in the first septuplet

state = tensor(Tminus, Tminus, Tminus, S)
state_t = basis_transformation(state, trans_matrix).unit()
print(state_t) # also lives in the first septuplet


# --- QUINTUPLETS

# We propose 
state = tensor(S, S, Tminus, Tminus)
state_t = basis_transformation(state, trans_matrix).unit()
print(state_t) # all zeros except one 1 at position 100
# this state belongs to the first quintuplet [98:103] (205 not included)

# --- TRIPLETS

# We propose 
state = tensor(S, S, S, Tminus)
state_t = basis_transformation(state, trans_matrix).unit()
state_t # all zeros except one 1 at position 31
# this state belongs to the sixth triplet [29:32] (32 not included)

state = tensor(Tminus, S, S, S)
state_t = basis_transformation(state, trans_matrix).unit()
state_t # belongs to many subspaces

state = tensor(S, Tminus, S, S)
state_t = basis_transformation(state, trans_matrix).unit()
state_t # belongs to many subspaces

state = tensor(S, S, Tminus, S)
state_t = basis_transformation(state, trans_matrix).unit()
state_t # belongs to many subspaces

state = tensor(S13T24, S, S)
state_t = basis_transformation(state, trans_matrix).unit()
state_t # belongs to many subspaces

state = tensor(T13S24, S, S)
state_t = basis_transformation(state, trans_matrix).unit()
state_t # belongs to many subspaces

state = tensor(S, S, S13T24)
state_t = basis_transformation(state, trans_matrix).unit()
state_t # belongs to many subspaces

state = tensor(S, S, T13S24)
state_t = basis_transformation(state, trans_matrix).unit()
state_t # belongs to many subspaces

state = tensor(S, S13T24, S)
state_t = basis_transformation(state, trans_matrix).unit()
state_t # belongs to many subspaces

state = tensor(S, T13S24, S)
state_t = basis_transformation(state, trans_matrix).unit()
state_t # belongs to many subspaces

state = tensor(S13S24, S, Tminus)
state_t = basis_transformation(state, trans_matrix).unit()
state_t

state = tensor(S, S13S24, Tminus)
state_t = basis_transformation(state, trans_matrix).unit()
state_t

state = tensor(S, Tminus, S13S24)
state_t = basis_transformation(state, trans_matrix).unit()
state_t

state = tensor(Tminus, S13S24, S)
state_t = basis_transformation(state, trans_matrix).unit()
state_t

state = tensor(S13S24, Tminus, S)
state_t = basis_transformation(state, trans_matrix).unit()
state_t

state = tensor(Tminus, S, S13S24)
state_t = basis_transformation(state, trans_matrix).unit()
state_t

# Having Tminus in front is close to first triplet
state = tensor(S13T24, S13S24)
state_t = basis_transformation(state, trans_matrix).unit()
state_t

state = tensor(T13S24, S13S24)
state_t = basis_transformation(state, trans_matrix).unit()
state_t

for index, element in enumerate(state_t):
    if element != 0.:
        print(index, element)
