#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 17:42:01 2023

@author: ceboncompte
"""

# We import the functions

from hamiltonian import hheis_general, ladder_exchanges, is_unitary
from use import matrix_plot, basis_transformation
from basis_matrix import coupled_matrix_gen

# Extra libraries

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# To create a Hamiltonian first you have to define the list of exchange couplings
# If you want to create a Heisenberg chain use only
J12 = 1
J23 = 2
J34 = 1

# Set the magnetic field

B = 0.5

list_of_Jij = np.array([J12, J23, J34])
spins = len(list_of_Jij) + 1 #always fixed

# To build the Hamiltonian you have to call

H_chain = hheis_general(list_of_Jij, spins, B)

# We print it

matrix_plot(H_chain)

# To build the basis transoformation matrix

trans_matrix = coupled_matrix_gen(spins)

# To apply it over on the Hamiltonian

H_coup = basis_transformation(H_chain, trans_matrix)

# We print it

matrix_plot(H_coup)

# Now you can do time evolution within a subspace, see the other files for more guidance

# If you want to create a Heisenberg ladder you have to define
J12, J34, J56 = 1, 1, 1
J13, J24, J35, J46 = 1.2, 1.2, 1.2, 1.2

Jij_vector = np.array([J12, 0, J34, 0, J56]) #every even entri here has to be zero
Jij_ladder = np.array([J13, J24, J35, J46])

H_ladder = hheis_general(Jij_vector, spins, B) + ladder_exchanges(Jij_ladder, spins)

# We do the same
matrix_plot(H_ladder)
#trans_matrix = coupled_matrix_gen(spins)
H_c = basis_transformation(H_ladder, trans_matrix)
matrix_plot(H_c)