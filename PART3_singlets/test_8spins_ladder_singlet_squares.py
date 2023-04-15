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
state = tensor(S, S, S, S)
state_t = basis_transformation(state, trans_matrix).unit()
state_t # all zeros except one 1 at position 0

# --------------------------------------------- FOR THE ENERGY PLOTS
number_iterations = 50 # number of different values of J we test
energy_tracker = np.zeros((number_iterations, 14)) # 14 refers to the number of states in the subspace
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

for J46 in values_J46:
    # We first want to create the arbitrary Hamiltonian and print the matrix
    Jij_vector = np.array([1, 0, 1, 0, 1, 0, 1])
    Jij_ladder = np.array([1, 1, J35zero - J46, J46, 1, 1])
    spins = len(Jij_vector) + 1
    
    H = hheis_general(Jij_vector, spins, B) + ladder_exchanges(Jij_ladder, spins)
    
    # We plot the matrix
#    matrix_plot(H)
    
    # We generate the basis-transformation matrix
    trans_matrix = coupled_matrix_gen(spins)
    
    # We finally basis transform and plot again the Hamiltonian matrix
    H_coup = basis_transformation(H, trans_matrix)
#    matrix_plot(H_coup)
    
    # We take the first septuplet and we get the energies of the states
    H_q = H_coup[0:14, 0:14]
    
    # --------------------------------------------- OSCILLATIONS
    # We transform this Hamiltonian into an H_channel (non time dependent)
    CH0 = H_channel(H_q)
    CH0.pulse.add_constant(2*np.pi*1.)
    
    # We solve the Schrodinger equation
    calculation = H_solver()
    calculation.add_channels(CH0)
    
    # Intitial state is the second state of the first triplet of the basis-transformation matrix, so we get its density matrix
    ket0 = basis(14, 0)
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

# we plot the energy of the eigenstates
plt.figure()
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[0,:], label ='E1')

plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[1,:], label ='E2')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[2,:], label ='E3')

plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[3,:], label ='E4')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[4,:], label ='E5')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[5,:], label ='E6')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[6,:], label ='E7')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[7,:], label ='E8')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[8,:], label ='E9')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[9,:], label ='E10')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[10,:], label ='E11')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[11,:], label ='E12')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[12,:], label ='E13')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[13,:], label ='E14')

plt.legend()
plt.xlabel('$J_{35} - J_{46}$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
plt.title('Energy of the singlets')
plt.show()

# we plot the differences of succesive eigenstate's energies
plt.figure()
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[1,:] - np.transpose(energy_tracker)[0,:], label ='E2 - E1')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[2,:] - np.transpose(energy_tracker)[1,:], label ='E3 - E2')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[3,:] - np.transpose(energy_tracker)[2,:], label ='E4 - E3')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[4,:] - np.transpose(energy_tracker)[3,:], label ='E5 - E4')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[5,:] - np.transpose(energy_tracker)[4,:], label ='E6 - E5')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[6,:] - np.transpose(energy_tracker)[5,:], label ='E7 - E6')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[7,:] - np.transpose(energy_tracker)[6,:], label ='E8 - E7')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[8,:] - np.transpose(energy_tracker)[7,:], label ='E9 - E8')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[9,:] - np.transpose(energy_tracker)[8,:], label ='E10 - E9')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[10,:] - np.transpose(energy_tracker)[9,:], label ='E11 - E10')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[11,:] - np.transpose(energy_tracker)[10,:], label ='E12 - E11')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[12,:] - np.transpose(energy_tracker)[11,:], label ='E13 - E12')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[13,:] - np.transpose(energy_tracker)[12,:], label ='E14 - E13')
plt.legend()
plt.xlabel('$J_{35} - J_{46}$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
plt.title('Differences of energy of the singlets')
plt.show()

# finally we colour plot the expect_tracker matrix data array and the amplitude_tracker for the fourier transform
x = t
y = np.linspace(J35zero-2*J46ini, J35zero-2*J46fin, number_iterations)
plt.pcolormesh(x, y, expect_tracker)
plt.title('Oscillations for the singlet subspace')
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