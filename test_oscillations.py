#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 16:19:40 2022

@author: ceboncompte
"""

from DM_solver.solver  import H_channel, H_solver
from DM_solver.pulse_generation.pulse import pulse
from DM_solver.utility.pauli import X,Y,Z

from qutip import *
from functions_n import heisenberg_hamiltonian, matrix_plot, coupled_basis_matrix, basis_transformation
from functions_n import normal_plot, Jij
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

# General input for simulation
B = 0
dimensions_expect_tracker = 1000
number_iterations = 100

# Range values of voltages for simulation
t12 = 8.5
t23 = 7.5
t34 = 11.9 #in microeV, these are fixed

"""
e12 = -2 #in mV
e23 = -1.8 #in mV
values = np.linspace(-7, -2.2, number_iterations) # values for e34 in mV

# We need to convert these voltage values into energy values (microeV)
converter = np.array([76, 81, 87, 84])
np.mean(converter[0:2])
epsil12 = e12 * np.mean(converter[0:2])
epsil23 = e23 * np.mean(converter[1:3])
values34 = values * np.mean(converter[2:4])
"""

e12 = 10 #in microeV
e23 = 0.2 #in microeV
values = np.linspace(12.5, -12.5, number_iterations) # values for e34 in microeV

# Calculation of the rest of inputs
J12 = Jij(e12, t12)
J23 = Jij(e23, t23) # in microeV

rows_same = int(dimensions_expect_tracker/number_iterations) # number of rows with same content to make expect_tracker square
expect_tracker = np.zeros((dimensions_expect_tracker, dimensions_expect_tracker))
#fourier_tracker = np.zeros((dimensions_expect_tracker, dimensions_expect_tracker))
amplitude_tracker = np.zeros((100,100))
n = 0

for e34 in values:

    # ---------- BASIS TRANSFORMATION ----------
    
    # We fist get the Heisenberg Hamiltonian in the computational basis and print it as a figure
    #J12 = 1
    #J23 = 1
    #J34 = 1
    #B = 0
    J34 = Jij(e34, t34)
    
    q = rows_same
    print(J12, J12-J34, e34)
    
    H = heisenberg_hamiltonian(J12, J23, J34, B)
#    matrix_plot(H)
    
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
#    matrix_plot(H_block_Tplus)
    
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
    end_time = 5
    number_steps = dimensions_expect_tracker
    sample_rate = number_steps/end_time
    calculation.calculate(dm0, end_time = end_time, sample_rate = sample_rate)
    
    # calculate the expectatoin value of the matrix dm0 and plot
    # plus fourier transform
    dm0_expect = calculation.return_expectation_values(dm0)
    # q = rows_same # introduced above
    expect_tracker[q*n:q*(n+1),:] = dm0_expect[0]
    t = calculation.return_time()
    
    time_step = 1/sample_rate
    sig_fft = fftpack.fft(dm0_expect[0])
    amplitude = np.abs(sig_fft)
    power = amplitude**2
    angle = np.angle(sig_fft)
    sample_freq = fftpack.fftfreq(dm0_expect[0].size, d = time_step)
    amplitude[0] = 0 # we set the zero frequency amplitude
    amplitude_tracker[n,:] = amplitude[:100]
    n += 1
    
# finally we colour plot the expect_tracker matrix data array
plt.imshow(expect_tracker)
plt.xlabel('time (0 to 1000000)')
plt.ylabel('nol')
plt.colorbar()
plt.show()

plt.imshow(amplitude_tracker)
plt.xlabel('time (0 to 1000000)')
plt.ylabel('nol')
plt.colorbar()
plt.show()

# --------------------------- now we do the same but instead of varying the e12-e34 detunning we vary the e23 ---------------------------------

e12 = 10 #in microeV
e34 = 2.4 #in microeV
values = np.linspace(50, -12.5, number_iterations) # values for e23 in microeV

# Calculation of the rest of inputs
J12 = Jij(e12, t12)
J34 = Jij(e34, t34)
#J23 = Jij(e23, t23) # in microeV

rows_same = int(dimensions_expect_tracker/number_iterations) # number of rows with same content to make expect_tracker square
expect_tracker = np.zeros((dimensions_expect_tracker, dimensions_expect_tracker))
#fourier_tracker = np.zeros((dimensions_expect_tracker, dimensions_expect_tracker))
amplitude_tracker = np.zeros((100,100))
n = 0

for e23 in values:

    # ---------- BASIS TRANSFORMATION ----------
    
    # We fist get the Heisenberg Hamiltonian in the computational basis and print it as a figure
    #J12 = 1
    #J23 = 1
    #J34 = 1
    #B = 0
    J23 = Jij(e23, t23)
    
    print(J12, J23, e23)
    
    H = heisenberg_hamiltonian(J12, J23, J34, B)
#    matrix_plot(H)
    
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
#    matrix_plot(H_block_Tplus)
    
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
    end_time = 5
    number_steps = dimensions_expect_tracker
    sample_rate = number_steps/end_time
    calculation.calculate(dm0, end_time = end_time, sample_rate = sample_rate)
    
    # calculate the expectatoin value of the matrix dm0 and plot
    # plus fourier transform
    dm0_expect = calculation.return_expectation_values(dm0)
    # q = rows_same # introduced above
    expect_tracker[q*n:q*(n+1),:] = dm0_expect[0]
    t = calculation.return_time()
    
    time_step = 1/sample_rate
    sig_fft = fftpack.fft(dm0_expect[0])
    amplitude = np.abs(sig_fft)
    power = amplitude**2
    angle = np.angle(sig_fft)
    sample_freq = fftpack.fftfreq(dm0_expect[0].size, d = time_step)
    amplitude[0] = 0 # we set the zero frequency amplitude
    amplitude_tracker[n,:] = amplitude[:100]
    n += 1
    
# finally we colour plot the expect_tracker matrix data array
plt.imshow(expect_tracker)
plt.xlabel('time (0 to 1000000)')
plt.ylabel('nol')
plt.colorbar()
plt.show()

plt.imshow(amplitude_tracker)
plt.xlabel('time (0 to 1000000)')
plt.ylabel('nol')
plt.colorbar()
plt.show()