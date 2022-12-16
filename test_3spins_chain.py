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
np.array_equal(H_coup[0:2,0:2], H_coup[2:4,2:4])
# Quadruplet is degenerate in energy for 0 magnetic field
# We diagonalize the doublet subspace
H_doub = H_coup[0:2,0:2]
e_doub = sp.linalg.eigvalsh(H_doub)
matrix_plot(np.diag(e_doub))
# Therefore the ground state is in the doublet subspace and degenerate in 2 eigenvectors
# However magnetic field breaks degeneracy
# To characterize the homogeneous case here we have to look at the coherent oscillations
# when magnetic field is present

# We want to see the energies of the eigenstates for this case, so we set a low magnetic field (also works with 0) and
# compute-plot them for different values of J
# We also want to see the oscillationss of the input states for this subspace

# --------------------------------------------- FOR THE ENERGY PLOTS
number_iterations = 100 # number of different values of J we test
energy_tracker = np.zeros((number_iterations, 2)) # 2 (doublet) refers to the number of states in the subspace
J23ini = 1.5
J23fin = 0.5
values_J23 = np.linspace(J23ini, J23fin, number_iterations)
#values_J23 = np.linspace(2, 0, number_iterations)
n = 0

# --------------------------------------------- FOR THE OSCILLATIONS AND FOURIERS
end_time = 5 # time of the oscillations (in natural units)
dimensions_expect_tracker = 1000 # number of time steps
expect_tracker = np.zeros((number_iterations, dimensions_expect_tracker))
amplitude_tracker = np.zeros((number_iterations, dimensions_expect_tracker))
J12zero = 2 # tuning this value (and sweeping for long J12 range) you make sure J values cross at the exact value J12zero/2,
# so the value of J in which we obtain homogeneous coupling can be chosen

for J23 in values_J23:
    # We first want to create the arbitrary Hamiltonian and print the matrix
    Jij_vector = np.array([J12zero-J23, J23])
    spins = len(Jij_vector) + 1
    
    H = hheis_general(Jij_vector, spins, B)
    
    # We plot the matrix
#    matrix_plot(H)
    
    # We generate the basis-transformation matrix
    trans_matrix = coupled_matrix_gen(spins)
    
    # We finally basis transform and plot again the Hamiltonian matrix
    H_coup = basis_transformation(H, trans_matrix)
#    matrix_plot(H_coup)
    
    # We take the first doublet and we get the energies of the states
    H_doublet = H_coup[0:2, 0:2]
#    matrix_plot(H_doublet)
    
    # --------------------------------------------- OSCILLATIONS
    # We transform this Hamiltonian into an H_channel (non time dependent)
    CH0 = H_channel(H_doublet)
    CH0.pulse.add_constant(2*np.pi*1.) # natural units of f (1/time usually)
    
    # We solve the Schrodinger equation
    calculation = H_solver()
    calculation.add_channels(CH0)
    
    # Intitial state is the first state (first row) of the basis-transformation matrix
    # so we get its density matrix
    ket0 = basis(2, 0)
    dm0 = ket0 * ket0.dag()
    dm0 = np.array(dm0)
    
    # Calculate for end_time
    number_steps = dimensions_expect_tracker
    sample_rate = number_steps/end_time
    calculation.calculate(dm0, end_time = end_time, sample_rate = sample_rate)
    
    # Calculate the expectatoin value of the matrix dm0 and plot
    dm0_expect = calculation.return_expectation_values(dm0)
    expect_tracker[n,:] = dm0_expect[0]
    t = calculation.return_time()
    
    # Fourier ttransform
    time_step = 1/sample_rate
    sig_fft = sp.fftpack.fft(dm0_expect[0]) # we get the fft of the signal (dm0_expect)
    amplitude = np.abs(sig_fft)
    power = amplitude**2
    angle = np.angle(sig_fft)
    sample_freq = sp.fftpack.fftfreq(dm0_expect[0].size, d = time_step) # length dimensions_expect_tracker (1000)
    amplitude[0] = 0 # we set the zero frequency amplitude
    amplitude_tracker[n,:] = amplitude # amplitude has length of dimensions_expect_tracker (1000)

    # We diagonalize and obtain energy values
    ev = sp.linalg.eigvalsh(H_doublet)
    energy_tracker[n,:] = ev
    n+=1

# we plot the energy of the eigenstates
#plt.figure(figsize=(6,5))
plt.figure()
plt.plot(J12zero-2*np.linspace(J23ini, J23fin, number_iterations), np.transpose(energy_tracker)[0,:], label='E1')
plt.plot(J12zero-2*np.linspace(J23ini, J23fin, number_iterations), np.transpose(energy_tracker)[1,:], label='E2')
plt.xlabel('$J_{12} - J_{23}$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
plt.title('Energies of the doublet subspace')
plt.show()

# finally we colour plot the expect_tracker matrix data array and the amplitude_tracker for the fourier transform
x = t
y = np.linspace(J12zero-2*J23ini, J12zero-2*J23fin, number_iterations)
plt.pcolormesh(x, y, expect_tracker)
plt.title('Oscillations in the doublet subspace')
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{12} - J_{23}$ ($E_0$)')
plt.colorbar()
plt.show()

# to plot the whole frequency spectrum (with negative ones)
x = sample_freq
y = np.linspace(J12zero-2*J23ini, J12zero-2*J23fin, number_iterations)
plt.pcolormesh(x[:80], y, amplitude_tracker[:,:80])
plt.title('Fourier transform of the oscillations in the doublet subspace')
plt.xlabel('frequency ($f_0$)')
plt.ylabel('$J_{12} - J_{23}$ ($E_0$)')
plt.colorbar()
plt.show()