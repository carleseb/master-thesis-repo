#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 14:17:38 2022

@author: ceboncompte
"""
from DM_solver.solver  import H_channel, H_solver
from DM_solver.pulse_generation.pulse import pulse
from DM_solver.utility.pauli import X,Y,Z

from qutip import *
from functions import heisenberg_hamiltonian, matrix_plot, coupled_basis_matrix, basis_transformation, normal_plot
import numpy as np
import matplotlib.pyplot as plt

# ---------- BASIS TRANSFORMATION ----------

# We fist get the Heisenberg Hamiltonian in the computational basis and print it as a figure
J12 = 1
J23 = 1
J34 = 1
B = 0

H = heisenberg_hamiltonian(J12, J23, J34, B)
matrix_plot(H)

# Now we obtain the basis-transform matrix for going from the computational basis to the coupled basis
spins = 4
j1 = 1/2
j2 = 1/2
j3 = 1/2
j4 = 1/2 # This is not needed by now

trans_matrix = coupled_basis_matrix(spins)

# Finally we change basis the Hamiltonian and we plot it again
H_block = basis_transformation(H, trans_matrix)
matrix_plot(H_block)

# ---------- TIME EVOLUTION ----------

# We take the block diagonal Hamiltonian and we slice it
H_block_Tplus = np.array(H_block)[5:8,5:8] # Special for triplet
matrix_plot(H_block_Tplus)

# We transform this Hamiltonian into an H_channel (non time dependent)
CH0 = H_channel(H_block_Tplus)
CH0.pulse.add_constant(1.)

# We solve the Schrodinger equation
calculation = H_solver()
calculation.add_channels(CH0)

# Intitial state is np.array([0, 1, 0]) transposed, so we get its density matrix
ket0 = basis(3,1) # Special for triplet
dm0 = ket0 * ket0.dag()
dm0 = np.array(dm0)

# calculate for 100ns with time steps of 10ps
# calculation.calculate(dm0, end_time = 100e-9, sample_rate = 1e11)
# number_steps = 100e-9/10e-12

# calculate for 10s with time steps of 0.001s
end_time = 100
sample_rate = 10000
calculation.calculate(dm0, end_time = end_time, sample_rate = sample_rate)
number_steps = end_time*sample_rate

# calculate the expectatoin value of the matrix dm0 and plot
dm0_expect = calculation.return_expectation_values(dm0)
t = calculation.return_time()
normal_plot(t, dm0_expect[0])