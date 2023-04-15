#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 16:22:26 2022

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

# We propose 
state = tensor(S, S, S, Tminus)
state_t = basis_transformation(state, trans_matrix).unit()
state_t # all zeros except one 1 at position 31
# this state belongs to the sixth triplet [29:32] (32 not included)

# --------------------------------------------- FOR THE ENERGY PLOTS
number_iterations = 50 # number of different values of J we test
energy_tracker = np.zeros((number_iterations, 3)) # 3 (triplet) refers to the number of states in the subspace
J46ini = 1.5
J46fin = 0.5
values_J46 = np.linspace(J46ini, J46fin, number_iterations)
n = 0

# --------------------------------------------- FOR THE OSCILLATIONS AND FOURIERS
end_time = 100 # time of the oscillations (in natural units)
dimensions_expect_tracker = 100 # number of time steps
expect_tracker = np.zeros((number_iterations, dimensions_expect_tracker))
amplitude_tracker = np.zeros((number_iterations, dimensions_expect_tracker))
J35zero = 2

# We generate the basis-transformation matrix
trans_matrix = coupled_matrix_gen(spins)

for J46 in values_J46:
    print('Obtaining plots... %', n*100/number_iterations)
    # We first want to create the arbitrary Hamiltonian and print the matrix
    #Jij_vector = np.array([1, 0, 1, 0, 1, 0, 1])
    #Jij_ladder = np.array([1, 1, J35zero - J46, J46, 1, 1])
    
    Jij_vector = np.array([J35zero - J46, 0, J35zero - J46, 0, J46, 0, J46])
    Jij_ladder = np.array([J35zero - J46, J35zero - J46, 0.6, 0.45, J46, J46])
    spins = len(Jij_vector) + 1
    B = 0
    
    H = hheis_general(Jij_vector, spins, B) + ladder_exchanges(Jij_ladder, spins)
    
    # We plot the matrix
#    matrix_plot(H)
    
    # We finally basis transform and plot again the Hamiltonian matrix
    H_coup = basis_transformation(H, trans_matrix)
#    matrix_plot(H_coup)
    
    # We take the first septuplet and we get the energies of the states
    #H_q = H_coup[29:32, 29:32]
    #H_q = H_coup[32:35, 32:35]
    #H_q = H_coup[35:38, 35:38]
    #H_q = H_coup[38:41, 38:41]
    #H_q = H_coup[41:44, 41:44]
    #H_q = H_coup[44:47, 44:47]
    
    H_q = H_coup[14:17,14:17]
    
    # --------------------------------------------- OSCILLATIONS
    # We transform this Hamiltonian into an H_channel (non time dependent)
    CH0 = H_channel(H_q)
    CH0.pulse.add_constant(2*np.pi*1.)
    
    # We solve the Schrodinger equation
    calculation = H_solver()
    calculation.add_channels(CH0)
    
    # Intitial state is the second state of the first triplet of the basis-transformation matrix, so we get its density matrix
    ket0 = basis(3, 2)
#    ket0 = minus_0
    dm0 = ket0 * ket0.dag()
    dm0 = np.array(dm0)
    
    # calculate for end_time
    number_steps = dimensions_expect_tracker
    sample_rate = number_steps/end_time
    calculation.calculate(dm0, end_time = end_time, sample_rate = sample_rate)
    
    # calculate the expectatoin value of the matrix dm0 and plot
    # plus fourier transform
    dm0_expect = calculation.return_expectation_values(dm0)
    expect_tracker[n,:] = dm0_expect[0]
    t = calculation.return_time()
    
    # fourier ttransform
    time_step = 1/sample_rate
    sig_fft = sp.fftpack.fft(dm0_expect[0]) # we get the fft of the signal (dm0_expect)
    amplitude = np.abs(sig_fft)
    power = amplitude**2
    angle = np.angle(sig_fft)
    sample_freq = sp.fftpack.fftfreq(dm0_expect[0].size, d = time_step) # length dimensions_expect_tracker (1000)
    amplitude[0] = 0 # we set the zero frequency amplitude
    amplitude_tracker[n,:] = amplitude # amplitude has length of dimensions_expect_tracker (1000)
    
    # We diagonalize and obtain energy values
    ev = sp.linalg.eigvalsh(H_q)
    energy_tracker[n,:] = ev
    n+=1
    
n = 0

# we plot the energy of the eigenstates
plt.figure()
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[0,:], label ='first sept list (label 29)')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[1,:], label ='second sept list (label 30)')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[2,:], label ='third sept list (label 31)')
plt.legend()
plt.xlabel('$J_{35} - J_{46}$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
plt.title('Energy of the septuplets')
plt.show()

# we plot the differences of succesive eigenstate's energies
plt.figure()
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[1,:] - np.transpose(energy_tracker)[0,:], label ='second - first')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[2,:] - np.transpose(energy_tracker)[1,:], label ='third - second')
plt.legend()
plt.xlabel('$J_{35} - J_{46}$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
plt.title('Differences of energy of the septuplets')
plt.show()

# finally we colour plot the expect_tracker matrix data array and the amplitude_tracker for the fourier transform
x = t
y = np.linspace(J35zero-2*J46ini, J35zero-2*J46fin, number_iterations)
plt.pcolormesh(x, y, expect_tracker)
plt.title('Oscillations for the quintuplet subspace')
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{35} - J_{46}$ ($E_0$)')
plt.colorbar()
plt.show()

# to plot the whole frequency spectrum (with negative ones)
x = sample_freq
y = np.linspace(J35zero-2*J46ini, J35zero-2*J46fin, number_iterations)
plt.pcolormesh(x[:50], y, amplitude_tracker[:,:50])
plt.title('Fourier transfom of the oscillations')
plt.xlabel('frequency ($f_0$)')
plt.ylabel('$J_{23} - J_{45}$ ($E_0$)')
plt.colorbar()
plt.show()