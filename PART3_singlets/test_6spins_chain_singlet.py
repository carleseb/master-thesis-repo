#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 13:13:22 2022

@author: ceboncompte
"""

from DM_solver.solver  import H_channel, H_solver

from hamiltonian import hheis_general, heisenberg_hamiltonian_4, is_unitary
from use import matrix_plot, basis_transformation
from basis_matrix import coupled_matrix_gen

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# We first want to create the arbitrary Hamiltonian and print the matrix
Jij_vector = np.array([1, 1, 1, 1, 1])
B = 0.5
spins = len(Jij_vector) + 1

# We check the Hamiltonian is Hermitian
H = hheis_general(Jij_vector, spins, B)
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

# Future: look at energies target state (target state is the scenario where the Js have the values
# we want): it seems that here middle triplet shows equal spacing for all states for B=0.3
"""
# We get the eigenvalues
e = sp.linalg.eigvalsh(H_coup)
print(e)
"""

# We want to see the energies of the eigenstates for this case, so we set a low magnetic field (also works with 0) and
# compute-plot them for different values of J
# We also want to see the oscillationss of the input states for this subspace
# We compute the initial state (Tminus, S) in the computational basis and we basis-transform it

# --------------------------------------------- MUST CHANGE TO A 6 SPIN SUBSPACE (todo)

S = (basis(4,1) - basis(4,2)).unit()
T0 = (basis(4,1) + basis(4,2)).unit()
Tplus = basis(4,0)
Tminus = basis(4,3)
# -------------------------------------------------------------------------------------

state = tensor(S, S, S)
state_t = basis_transformation(state, trans_matrix).unit()
print(state_t[0:5])

# --------------------------------------------- FOR THE ENERGY PLOTS
number_iterations = 100 # number of different values of J we test
energy_tracker = np.zeros((number_iterations, 5)) # 5 (singlets) refers to the number of states in the subspace
J34ini = 1.5
J34fin = 0.5
values_J34 = np.linspace(J34ini, J34fin, number_iterations)
n = 0

# --------------------------------------------- FOR THE OSCILLATIONS AND FOURIERS
end_time = 100 # time of the oscillations (in natural units)
dimensions_expect_tracker = 1000 # number of time steps
expect_tracker = np.zeros((number_iterations, dimensions_expect_tracker))
amplitude_tracker = np.zeros((number_iterations, dimensions_expect_tracker))
J12zero = 2

for J34 in values_J34:
    # We first want to create the arbitrary Hamiltonian and print the matrix
#    Jij_vector = np.array([1, J12zero - J34, 1, J34, 1])
#    Jij_vector = np.array([J12zero - J34, 0.2, J34, 0.2, J12zero - J34])
#    Jij_vector = np.array([J12zero - J34, 0.2, J34, 0, 0])
    Jij_vector = np.array([1, 1, J34, 1, 1])
    spins = len(Jij_vector) + 1
    
    H = hheis_general(Jij_vector, spins, B)
    
    # We plot the matrix
#    matrix_plot(H)
    
    # We generate the basis-transformation matrix
    trans_matrix = coupled_matrix_gen(spins)
    
    # We finally basis transform and plot again the Hamiltonian matrix
    H_coup = basis_transformation(H, trans_matrix)
#    matrix_plot(H_coup)
    
    # We have a subspace for 5 singlets, 9 triplets, 5 quintuplets and 1 septuplet
    # We take the first quintuplet and we get the energies of the states
    H_q = H_coup[0:5, 0:5]
    
    # --------------------------------------------- OSCILLATIONS
    # We transform this Hamiltonian into an H_channel (non time dependent)
    CH0 = H_channel(H_q)
    CH0.pulse.add_constant(2*np.pi*1.)
    
    # We solve the Schrodinger equation
    calculation = H_solver()
    calculation.add_channels(CH0)
    
    # Intitial state is the second state of the first triplet of the basis-transformation matrix, so we get its density matrix
    ket0 = basis(5, 0) # State SSS
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
plt.plot(J12zero-2*np.linspace(J34ini, J34fin, number_iterations), np.transpose(energy_tracker)[0,:], label ='E1')
plt.plot(J12zero-2*np.linspace(J34ini, J34fin, number_iterations), np.transpose(energy_tracker)[1,:], label ='E2')
plt.plot(J12zero-2*np.linspace(J34ini, J34fin, number_iterations), np.transpose(energy_tracker)[2,:], label ='E3')
plt.plot(J12zero-2*np.linspace(J34ini, J34fin, number_iterations), np.transpose(energy_tracker)[3,:], label ='E4')
plt.plot(J12zero-2*np.linspace(J34ini, J34fin, number_iterations), np.transpose(energy_tracker)[4,:], label ='E5')
plt.legend()
plt.xlabel('$J_{23} - J_{45}$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
plt.title('Energies of the singlet subspace')
plt.show()

# we plot the differences of succesive eigenstate's energies
plt.plot(J12zero-2*np.linspace(J34ini, J34fin, number_iterations), np.transpose(energy_tracker)[1,:] - np.transpose(energy_tracker)[0,:], label ='E2 - E1')
plt.plot(J12zero-2*np.linspace(J34ini, J34fin, number_iterations), np.transpose(energy_tracker)[2,:] - np.transpose(energy_tracker)[1,:], label ='E3 - E2')
plt.plot(J12zero-2*np.linspace(J34ini, J34fin, number_iterations), np.transpose(energy_tracker)[3,:] - np.transpose(energy_tracker)[2,:], label ='E4 - E3')
plt.plot(J12zero-2*np.linspace(J34ini, J34fin, number_iterations), np.transpose(energy_tracker)[4,:] - np.transpose(energy_tracker)[3,:], label ='E5 - E4')
plt.legend()
plt.xlabel('$J_{23} - J_{45}$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
plt.title('Energy differences of adjacent states of the singlet subspace')
plt.show()

# finally we colour plot the expect_tracker matrix data array and the amplitude_tracker for the fourier transform
x = t
y = np.linspace(J12zero-2*J34ini, J12zero-2*J34fin, number_iterations)
plt.pcolormesh(x, y, expect_tracker)
plt.title('Oscillations in the singlet subspace')
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{23} - J_{45}$ ($E_0$)')
plt.colorbar()
plt.show()

# to plot the whole frequency spectrum (with negative ones)
x = sample_freq
y = np.linspace(J12zero-2*J34ini, J12zero-2*J34fin, number_iterations)
plt.pcolormesh(x[:100], y, amplitude_tracker[:,:100])
plt.title('Fourier transform of the oscillations in the singlet subspace')
plt.xlabel('frequency ($f_0$)')
plt.ylabel('$J_{23} - J_{45}$ ($E_0$)')
plt.colorbar()
plt.show()

# Now we take the value for which we observe minimum energy gap between bot triplets (and homogeneous exchange)
# and we fix the J12 and J34 values, varying the J23 to obtain same value for all exchanges, we will see condition in energy

# --------------------------------------------- FOR THE ENERGY PLOTS
number_iterations = 100 # number of different values of J we test
energy_tracker_ = np.zeros((number_iterations, 5)) # 5 (singlets) refers to the number of states in the subspace
J34ini = 1.5
J34fin = 0.5
values_J34 = np.linspace(J34ini, J34fin, number_iterations)
#values_J34 = np.linspace(2, 0, number_iterations)
n = 0

# --------------------------------------------- FOR THE OSCILLATIONS AND FOURIERS
end_time = 30 # time of the oscillations (in natural units)
dimensions_expect_tracker = 1000 # number of time steps
expect_tracker = np.zeros((number_iterations, dimensions_expect_tracker))
amplitude_tracker = np.zeros((number_iterations, dimensions_expect_tracker))

for J34 in values_J34:
    # We first want to create the arbitrary Hamiltonian and print the matrix
    Jij_vector = np.array([1, 1, J34, 1, 1])
    B = 0
    spins = len(Jij_vector) + 1
    
    H = hheis_general(Jij_vector, spins, B)
    
    # We plot the matrix
#    matrix_plot(H)
    
    # We generate the basis-transformation matrix
    trans_matrix = coupled_matrix_gen(spins)
    
    # We finally basis transform and plot again the Hamiltonian matrix
    H_coup = basis_transformation(H, trans_matrix)
#    matrix_plot(H_coup)
    
    # We take the first triplet and we get the energies of the states
    H_q = H_coup[0:4, 0:4]
    
    # --------------------------------------------- OSCILLATIONS
    # We transform this Hamiltonian into an H_channel (non time dependent)
    CH0 = H_channel(H_q)
    CH0.pulse.add_constant(2*np.pi*1.)
    
    # We solve the Schrodinger equation
    calculation = H_solver()
    calculation.add_channels(CH0)
    
    # Intitial state is the second state of the first triplet of the basis-transformation matrix, so we get its density matrix
    ket0 = basis(5, 0)
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
    sample_freq = sp.fftpack.fftfreq(dm0_expect[0].size, d = time_step)
    # sample_freq goes from  0 to (1/end_time)*(dimensions_expect_tracker/2) and then from negative
    # of (1/end_time)*(dimensions_expect_tracker/2) back to zero
    amplitude[0] = 0 # we set the zero frequency amplitude
    amplitude_tracker[n,:] = amplitude
    
    # We diagonalize and obtain energy values
    ev = sp.linalg.eigvalsh(H_q)
    energy_tracker_[n,:] = ev
    n+=1

# we plot the energy of the eigenstates
plt.figure()
plt.plot(np.linspace(J34ini, J34fin, number_iterations), np.transpose(energy_tracker_)[0,:], label ='first q list (label 32)')
plt.plot(np.linspace(J34ini, J34fin, number_iterations), np.transpose(energy_tracker_)[1,:], label ='second q list (label 33)')
plt.plot(np.linspace(J34ini, J34fin, number_iterations), np.transpose(energy_tracker_)[2,:], label ='third q list (label 34)')
plt.plot(np.linspace(J34ini, J34fin, number_iterations), np.transpose(energy_tracker_)[3,:], label ='fourth q list (label 35)')
plt.plot(np.linspace(J34ini, J34fin, number_iterations), np.transpose(energy_tracker_)[4,:], label ='fifth q list (label 36)')
plt.legend()
plt.xlabel('$J_{34}$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
plt.title('Energy of the singlets')
plt.show()

# we plot the differences of succesive eigenstate's energies
plt.plot(np.linspace(J34ini, J34fin, number_iterations), np.transpose(energy_tracker_)[1,:] - np.transpose(energy_tracker_)[0,:], label ='second - first')
plt.plot(np.linspace(J34ini, J34fin, number_iterations), np.transpose(energy_tracker_)[2,:] - np.transpose(energy_tracker_)[1,:], label ='third - second')
plt.plot(np.linspace(J34ini, J34fin, number_iterations), np.transpose(energy_tracker_)[3,:] - np.transpose(energy_tracker_)[2,:], label ='fourht - third')
plt.plot(np.linspace(J34ini, J34fin, number_iterations), np.transpose(energy_tracker_)[4,:] - np.transpose(energy_tracker_)[3,:], label ='fifth - fourth')
plt.legend()
plt.xlabel('$J_{34}$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
plt.title('Differences of energy of the singlet')
plt.show()

# finally we colour plot the expect_tracker matrix data array and the amplitude_tracker for the fourier transform
x = t
y = np.linspace(J34ini, J34fin, number_iterations)
plt.pcolormesh(x, y, expect_tracker)
plt.title('Oscillations for the singlet subspace')
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{34}$ ($E_0$)')
plt.colorbar()
plt.show()

# to plot the whole frequency spectrum (with negative ones)
x = sample_freq
y = np.linspace(J34ini, J34fin, number_iterations)
plt.pcolormesh(x[:100], y, amplitude_tracker[:,:100]) # the number we cut frequencies has to be
# maximum dimensions_expect_tracker/2
plt.title('Fourier transfom of the oscillations')
plt.xlabel('frequency ($f_0$)')
plt.ylabel('$J_{34}$ ($E_0$)')
plt.colorbar()
plt.show()