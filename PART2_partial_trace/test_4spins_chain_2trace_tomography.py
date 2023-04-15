#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 14:50:51 2022

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
Jij_vector = np.array([1, 1, 1])
B = 0.5
spins = len(Jij_vector) + 1

# We check the Hamiltonian is Hermitian
H = hheis_general(Jij_vector, spins, B)
H.check_herm()
H

# We plot the matrix
matrix_plot(H)

# We first want to check if the Hamiltonian that our general function yeilds is the same as we expect
# from a 3 spins Heisenberg Hamiltonian
H4 = heisenberg_hamiltonian_4(1, 0.2, 1, B)
H == H4
# We see that for any combination of Js and B we get exactly the same matrix

# We generate the basis-transformation matrix
trans_matrix = coupled_matrix_gen(spins)
matrix_plot(trans_matrix) # The basis states are rows
print(trans_matrix[0,:]) # first basis state
print(trans_matrix[3,:]) # basis state we evolve (1/root6, 2/root6 combinations)
print(trans_matrix[2,:])
print(trans_matrix[4,:]) 

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

# We want to see the energies of the eigenstates for this case, so we set a low magnetic field (also works with 0) and
# compute-plot them for different values of J
# We also want to see the oscillationss of the input states for this subspace
# We compute the initial state (Tminus, S) in the computational basis and we basis-transform it
S = (basis(4,1) - basis(4,2)).unit()
T0 = (basis(4,1) + basis(4,2)).unit()
Tplus = basis(4,0)
Tminus = basis(4,3)

# We build the general pair-product states
TM0 = tensor(S,Tminus)
TM1 = tensor(Tminus,S)
TM2 = (tensor(T0,Tminus) - tensor(Tminus,T0)).unit()

# We generate the basis-transformation matrix and we transform the states
trans_matrix = coupled_matrix_gen(spins)
TM0_t = basis_transformation(TM0, trans_matrix)
TM1_t = basis_transformation(TM1, trans_matrix).unit()
TM2_t = basis_transformation(TM2, trans_matrix).unit()
sup = (TM1_t + TM2_t).unit()
print(np.array(TM0_t), np.array(TM1_t), np.array(TM2_t))
minus_0 = TM0_t.eliminate_states([0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
print(np.array(minus_0))

## -------------------------------------------- GELL-MAN MATRICES
l1 = Qobj(np.array([[0, 1, 0],[1, 0, 0],[0, 0, 0]]))
l2 = Qobj(np.array([[0, -1j, 0],[1j, 0, 0],[0, 0, 0]]))
l3 = Qobj(np.array([[1, 0, 0],[0, -1, 0],[0, 0, 0]]))
l4 = Qobj(np.array([[0, 0, 1],[0, 0, 0],[1, 0, 0]]))

l5 = Qobj(np.array([[0, 0, -1j],[0, 0, 0],[1j, 0, 0]]))
l6 = Qobj(np.array([[0, 0, 0],[0, 0, 1],[0, 1, 0]]))
l7 = Qobj(np.array([[0, 0, 0],[0, 0, -1j],[0, 1j, 0]]))
l8 = Qobj(np.array([[1, 0, 0],[0, 1, 0],[0, 0, -2]])*(1/np.sqrt(3)))





# HERE WE START THE FIRST PART OF THE PROTOCOL

# We sweep antisimetrically J12 and J34

# --------------------------------------------- FOR THE ENERGY PLOTS
number_iterations = 100 # number of different values of J we test
energy_tracker = np.zeros((number_iterations, 3)) # 3 (triplet) refers to the number of states in the subspace
J34ini = 1.5
J34fin = 0.5
values_J34 = np.linspace(J34ini, J34fin, number_iterations)
#values_J34 = np.linspace(2, 0, number_iterations)
n = 0

# --------------------------------------------- FOR THE OSCILLATIONS AND FOURIERS
end_time = 30 # time of the oscillations (in natural units)
dimensions_expect_tracker = 1000 # number of time steps #BACK TO 1000!!!!!!1
expect_tracker = np.zeros((number_iterations, dimensions_expect_tracker))
et1 = np.zeros((number_iterations, dimensions_expect_tracker))
et2 = np.zeros((number_iterations, dimensions_expect_tracker))
et3 = np.zeros((number_iterations, dimensions_expect_tracker))
et4 = np.zeros((number_iterations, dimensions_expect_tracker))
et5 = np.zeros((number_iterations, dimensions_expect_tracker))
et6 = np.zeros((number_iterations, dimensions_expect_tracker))
et7 = np.zeros((number_iterations, dimensions_expect_tracker))
et8 = np.zeros((number_iterations, dimensions_expect_tracker))
density_matrices = np.zeros((number_iterations, dimensions_expect_tracker, 3, 3))
# because density matrices here are 3x3 matrices
amplitude_tracker = np.zeros((number_iterations, dimensions_expect_tracker))
J12zero = 2

for J34 in values_J34:
    # We first want to create the arbitrary Hamiltonian and print the matrix
    Jij_vector = np.array([J12zero-J34, 0.6, J34])
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
    H_triplet = H_coup[2:5, 2:5]
    
    # --------------------------------------------- OSCILLATIONS
    # We transform this Hamiltonian into an H_channel (non time dependent)
    CH0 = H_channel(H_triplet)
    CH0.pulse.add_constant(2*np.pi*1.)
    
    # We solve the Schrodinger equation
    calculation = H_solver()
    calculation.add_channels(CH0)
    
    # Intitial state is the second state of the first triplet of the basis-transformation matrix, so we get its density matrix
#    ket0 = basis(3, 1)
    ket0 = minus_0
    dm0 = ket0 * ket0.dag()
    dm0 = np.array(dm0)
    
    # calculate for end_time
    number_steps = dimensions_expect_tracker
    sample_rate = number_steps/end_time
    calculation.calculate(dm0, end_time = end_time, sample_rate = sample_rate)
    
    # calculate the expectatoin value of the matrix dm0 and plot
    # plus fourier transform
    # also calculate the expectation value of the Gell-Mann matrices
    dm0_expect = calculation.return_expectation_values(dm0)
    el1 = calculation.return_expectation_values(l1)
    el2 = calculation.return_expectation_values(l2)
    el3 = calculation.return_expectation_values(l3)
    el4 = calculation.return_expectation_values(l4)
    el5 = calculation.return_expectation_values(l5)
    el6 = calculation.return_expectation_values(l6)
    el7 = calculation.return_expectation_values(l7)
    el8 = calculation.return_expectation_values(l8)
    
    expect_tracker[n,:] = dm0_expect[0]
    et1[n,:] = el1[0]
    et2[n,:] = el2[0]
    et3[n,:] = el3[0]
    et4[n,:] = el4[0]
    et5[n,:] = el5[0]
    et6[n,:] = el6[0]
    et7[n,:] = el7[0]
    et8[n,:] = el8[0]
    
    #We obtain the density matrix of the state and we store them in a x = t = 1000 times
    # y = number_iterations = 100 matrix
    for k in range(dimensions_expect_tracker):
        e11 = 1+(np.sqrt(3)/2)*(et8[n,k]+np.sqrt(3)*et3[n,k])
        e12 = (3/2)*(et1[n,k]-1j*et2[n,k])
        e13 = (3/2)*(et4[n,k]-1j*et5[n,k])
        e21 = (3/2)*(et1[n,k]+1j*et2[n,k])
        e22 = 1+(np.sqrt(3)/2)*(et8[n,k]-np.sqrt(3)*et3[n,k])
        e23 = (3/2)*(et6[n,k]-1j*et7[n,k])
        e31 = (3/2)*(et4[n,k]+1j*et5[n,k])
        e32 = (3/2)*(et6[n,k]+1j*et7[n,k])
        e33 = 1-np.sqrt(3)*et8[n,k]
        density_matrices[n,k,:,:] = (1/3)*Qobj(np.array([[e11, e12, e13],[e21, e22, e23],[e31, e32, e33]]))
    
    t = calculation.return_time()
#    u = calculation.get_unitary()
#    print(u)
    
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
    ev = sp.linalg.eigvalsh(H_triplet)
    energy_tracker[n,:] = ev
    n+=1

# we plot the energy of the eigenstates
plt.figure()
'''
plt.plot(J12zero-2*np.linspace(J34ini, J34fin, number_iterations), np.transpose(energy_tracker)[0,:], label ='first triplet list (label 2)')
plt.plot(J12zero-2*np.linspace(J34ini, J34fin, number_iterations), np.transpose(energy_tracker)[1,:], label ='second triplet list (label 3)')
plt.plot(J12zero-2*np.linspace(J34ini, J34fin, number_iterations), np.transpose(energy_tracker)[2,:], label ='third triplet list (label 4)')
'''
plt.plot(J12zero-2*np.linspace(J34ini, J34fin, number_iterations), np.transpose(energy_tracker)[0,:], label ='E1')
plt.plot(J12zero-2*np.linspace(J34ini, J34fin, number_iterations), np.transpose(energy_tracker)[1,:], label ='E2')
plt.plot(J12zero-2*np.linspace(J34ini, J34fin, number_iterations), np.transpose(energy_tracker)[2,:], label ='E3')
plt.legend()
plt.xlabel('$J_{12} - J_{34}$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
plt.title('Energies of the triplet subspace')
plt.show()

# finally we colour plot the expect_tracker matrix data array and the amplitude_tracker for the fourier transform
x = t
y = np.linspace(J12zero-2*J34ini, J12zero-2*J34fin, number_iterations)
plt.pcolormesh(x, y, expect_tracker)
plt.title('Oscillations in the triplet subspace')
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{12} - J_{34}$ ($E_0$)')
plt.colorbar()
plt.show()

# to plot the whole frequency spectrum (with negative ones)
x = sample_freq
y = np.linspace(J12zero-2*J34ini, J12zero-2*J34fin, number_iterations)
plt.pcolormesh(x[:80], y, amplitude_tracker[:,:80])
plt.title('Fourier transform of the oscillations in the triplet subspace')
plt.xlabel('frequency ($f_0$)')
plt.ylabel('$J_{12} - J_{34}$ ($E_0$)')
plt.colorbar()
plt.show()

# now we want to store the value of the traces and plot them over time
trace_dm_tracker =  np.zeros((number_iterations, dimensions_expect_tracker))
trace_reduced_tracker = np.zeros((number_iterations, dimensions_expect_tracker))
expect_singlet_tracker = np.zeros((number_iterations, dimensions_expect_tracker))
singlet = (tensor(basis(2,0), basis(2,1)) - tensor(basis(2,1), basis(2,0))).unit()
singlet_dm = singlet*singlet.dag()
expect_triplet_tracker = np.zeros((number_iterations, dimensions_expect_tracker))
triplet = (tensor(basis(2,1), basis(2,1)))
triplet_dm = triplet*triplet.dag()
amplitude_tracker_singlet = np.zeros((number_iterations, dimensions_expect_tracker))
amplitude_tracker_triplet = np.zeros((number_iterations, dimensions_expect_tracker))
amplitude_tracker_trace = np.zeros((number_iterations, dimensions_expect_tracker))

for l in range(number_iterations):
    print('reconstructing dm...')
    for k in range(dimensions_expect_tracker):
        # we get the traces of the squares of the density matrices
        m = density_matrices[l,k,:,:]
        m_pad = np.pad(m,((2,11),(2,11)))
        m_pad = Qobj(m_pad)
        #print(m_pad.norm()) # this norm is correct
        W = Qobj(trans_matrix).dag()
        finaldm = basis_transformation(m_pad, W)
        #print(finaldm.norm()) # this norm is correct
        trace_dm_tracker[l,k] = finaldm.tr()
        # now we want to trace out the information of spins 3 and 4
        finaldm.dims =  [[2, 2, 2, 2], [2, 2, 2, 2]]
        dm12 = finaldm.ptrace([0, 1]) # We keep subspaces 0 and 1, so we trace 2 and 3
        #print(dm12.norm()) # this norm is correct
        trace_reduced_tracker[l,k] = (dm12*dm12).tr()
        
        # we get the expectation value of singlet in spins 1 and 2
        expect_singlet_tracker[l,k] = (dm12*singlet_dm).tr()
        expect_triplet_tracker[l,k] = (dm12*triplet_dm).tr()
        
    # fourier transform of the trace
    time_step = 1/sample_rate
    sig_fft_trace = sp.fftpack.fft(trace_reduced_tracker[l,:])
    amplitude_trace = np.abs(sig_fft_trace)
    sample_freq_trace = sp.fftpack.fftfreq(trace_reduced_tracker[l,:].size, d = time_step)
    amplitude_trace[0] = 0 # we set the zero frequency amplitude
    amplitude_tracker_trace[l,:] = amplitude_trace
    
    # fourier transform of the probabilities
    time_step = 1/sample_rate
    sig_fft_singlet = sp.fftpack.fft(expect_singlet_tracker[l,:]) # we get the fft of the signal
    sig_fft_triplet = sp.fftpack.fft(expect_triplet_tracker[l,:])
    amplitude_singlet = np.abs(sig_fft_singlet)
    amplitude_triplet = np.abs(sig_fft_triplet)
    sample_freq_singlet = sp.fftpack.fftfreq(expect_singlet_tracker[l,:].size, d = time_step)
    sample_freq_triplet = sp.fftpack.fftfreq(expect_triplet_tracker[l,:].size, d = time_step)
    amplitude_singlet[0] = 0 # we set the zero frequency amplitude
    amplitude_triplet[0] = 0
    amplitude_tracker_singlet[l,:] = amplitude_singlet
    amplitude_tracker_triplet[l,:] = amplitude_triplet

print('done')
# we plot
x = t
y = np.linspace(J12zero-2*J34ini, J12zero-2*J34fin, number_iterations)
#plt.pcolormesh(x, y, trace_dm_tracker)
plt.pcolormesh(x, y, trace_reduced_tracker)
plt.title('Oscillations of the trace of the reduced density matrix squared') #spins 1 and 2
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{12} - J_{34}$ ($E_0$)')
plt.colorbar()
plt.show()

# the fourier of the reduced osicllations (trace)
x = sample_freq_trace
y = np.linspace(J12zero-2*J34ini, J12zero-2*J34fin, number_iterations)
plt.pcolormesh(x[:80], y, amplitude_tracker_trace[:,:80])
plt.title('Fourier transform of the oscillations of the trace of the reduced density matrix squared')
plt.xlabel('frequency ($f_0$)')
plt.ylabel('$J_{12} - J_{34}$ ($E_0$)')
plt.colorbar()
plt.show()

x = t
y = np.linspace(J12zero-2*J34ini, J12zero-2*J34fin, number_iterations)
#plt.pcolormesh(x, y, trace_dm_tracker)
plt.pcolormesh(x, y, expect_singlet_tracker)
plt.title('Oscillations of the reduced system (proba singlet)') #spins 1 and 2
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{12} - J_{34}$ ($E_0$)')
plt.colorbar()
plt.show()

# the fourier of the reduced osicllations (singlet)
x = sample_freq_singlet
y = np.linspace(J12zero-2*J34ini, J12zero-2*J34fin, number_iterations)
plt.pcolormesh(x[:80], y, amplitude_tracker_singlet[:,:80])
plt.title('Fourier transform of the oscillations of the reduced singlet')
plt.xlabel('frequency ($f_0$)')
plt.ylabel('$J_{12} - J_{34}$ ($E_0$)')
plt.colorbar()
plt.show()

x = t
y = np.linspace(J12zero-2*J34ini, J12zero-2*J34fin, number_iterations)
#plt.pcolormesh(x, y, trace_dm_tracker)
plt.pcolormesh(x, y, expect_triplet_tracker)
plt.title('Oscillations of the reduced system (proba triplet)') #spins 1 and 2
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{12} - J_{34}$ ($E_0$)')
plt.colorbar()
plt.show()

# the fourier of the reduced osicllations (triplet)
x = sample_freq_triplet
y = np.linspace(J12zero-2*J34ini, J12zero-2*J34fin, number_iterations)
plt.pcolormesh(x[:80], y, amplitude_tracker_triplet[:,:80])
plt.title('Fourier transform of the oscillations of the reduced triplet')
plt.xlabel('frequency ($f_0$)')
plt.ylabel('$J_{12} - J_{34}$ ($E_0$)')
plt.colorbar()
plt.show()





# HERE WE START THE SECOND PART OF THE PROTOCOL

# Now we take the value for which we observe minimum energy gap between bot triplets (and homogeneous exchange)
# and we fix the J12 and J34 values, varying the J23 to obtain same value for all exchanges, we will see condition in energy

# --------------------------------------------- FOR THE ENERGY PLOTS
number_iterations = 100 # number of different values of J we test
energy_tracker_ = np.zeros((number_iterations, 3)) # 3 (triplet) refers to the number of states in the subspace
J23ini = 1.5
J23fin = 0.5
values_J23 = np.linspace(J23ini, J23fin, number_iterations)
#values_J23 = np.linspace(2, 0, number_iterations)
n = 0

# --------------------------------------------- FOR THE OSCILLATIONS AND FOURIERS
end_time = 30 # time of the oscillations (in natural units)
dimensions_expect_tracker = 1000 # number of time steps
expect_tracker = np.zeros((number_iterations, dimensions_expect_tracker))
et1 = np.zeros((number_iterations, dimensions_expect_tracker))
et2 = np.zeros((number_iterations, dimensions_expect_tracker))
et3 = np.zeros((number_iterations, dimensions_expect_tracker))
et4 = np.zeros((number_iterations, dimensions_expect_tracker))
et5 = np.zeros((number_iterations, dimensions_expect_tracker))
et6 = np.zeros((number_iterations, dimensions_expect_tracker))
et7 = np.zeros((number_iterations, dimensions_expect_tracker))
et8 = np.zeros((number_iterations, dimensions_expect_tracker))
density_matrices = np.zeros((number_iterations, dimensions_expect_tracker, 3, 3))
# because density matrices here are 3x3 matrices
amplitude_tracker = np.zeros((number_iterations, dimensions_expect_tracker))

for J23 in values_J23:
    # We first want to create the arbitrary Hamiltonian and print the matrix
    # However, now we want to trace out the information of the two right qubits
    # and only read out the two left qubits
    Jij_vector = np.array([2-1, J23, 1])
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
    H_triplet = H_coup[2:5, 2:5]
    
    # --------------------------------------------- OSCILLATIONS
    # We transform this Hamiltonian into an H_channel (non time dependent)
    CH0 = H_channel(H_triplet)
    CH0.pulse.add_constant(2*np.pi*1.)
    
    # We solve the Schrodinger equation
    calculation = H_solver()
    calculation.add_channels(CH0)
    
    # Intitial state is the second state of the first triplet of the basis-transformation matrix, so we get its density matrix
#    ket0 = basis(3, 1)
    ket0 = minus_0
    dm0 = ket0 * ket0.dag()
    dm0 = np.array(dm0)
    
    # calculate for end_time
    number_steps = dimensions_expect_tracker
    sample_rate = number_steps/end_time
    calculation.calculate(dm0, end_time = end_time, sample_rate = sample_rate)
    
    # calculate the expectatoin value of the matrix dm0 and plot
    # plus fourier transform
    # also calculate the expectation value of the Gell-Mann matrices
    dm0_expect = calculation.return_expectation_values(dm0)
    el1 = calculation.return_expectation_values(l1)
    el2 = calculation.return_expectation_values(l2)
    el3 = calculation.return_expectation_values(l3)
    el4 = calculation.return_expectation_values(l4)
    el5 = calculation.return_expectation_values(l5)
    el6 = calculation.return_expectation_values(l6)
    el7 = calculation.return_expectation_values(l7)
    el8 = calculation.return_expectation_values(l8)
    
    expect_tracker[n,:] = dm0_expect[0]
    et1[n,:] = el1[0]
    et2[n,:] = el2[0]
    et3[n,:] = el3[0]
    et4[n,:] = el4[0]
    et5[n,:] = el5[0]
    et6[n,:] = el6[0]
    et7[n,:] = el7[0]
    et8[n,:] = el8[0]
    
    #We obtain the density matrix of the state and we store them in a x = t = 1000 times
    # y = number_iterations = 100 matrix
    for k in range(dimensions_expect_tracker):
        e11 = 1+(np.sqrt(3)/2)*(et8[n,k]+np.sqrt(3)*et3[n,k])
        e12 = (3/2)*(et1[n,k]-1j*et2[n,k])
        e13 = (3/2)*(et4[n,k]-1j*et5[n,k])
        e21 = (3/2)*(et1[n,k]+1j*et2[n,k])
        e22 = 1+(np.sqrt(3)/2)*(et8[n,k]-np.sqrt(3)*et3[n,k])
        e23 = (3/2)*(et6[n,k]-1j*et7[n,k])
        e31 = (3/2)*(et4[n,k]+1j*et5[n,k])
        e32 = (3/2)*(et6[n,k]+1j*et7[n,k])
        e33 = 1-np.sqrt(3)*et8[n,k]
        density_matrices[n,k,:,:] = (1/3)*Qobj(np.array([[e11, e12, e13],[e21, e22, e23],[e31, e32, e33]]))
    
    t = calculation.return_time()
#    u = calculation.get_unitary()
#    print(u) # not being used in this script, see unitary one
    
    # fourier transform
    time_step = 1/sample_rate
    sig_fft = sp.fftpack.fft(dm0_expect[0]) # we get the fft of the signal (dm0_expect)
    amplitude = np.abs(sig_fft)
    power = amplitude**2
    angle = np.angle(sig_fft)
    sample_freq = sp.fftpack.fftfreq(dm0_expect[0].size, d = time_step)
    amplitude[0] = 0 # we set the zero frequency amplitude
    amplitude_tracker[n,:] = amplitude
    
    # We diagonalize and obtain energy values
    ev = sp.linalg.eigvalsh(H_triplet)
    energy_tracker_[n,:] = ev
    n+=1

# we plot the energy of the eigenstates
plt.figure()
'''
plt.plot(np.linspace(J34ini, J34fin, number_iterations), np.transpose(energy_tracker_)[0,:], label ='first triplet list (label 2)')
plt.plot(np.linspace(J34ini, J34fin, number_iterations), np.transpose(energy_tracker_)[1,:], label ='second triplet list (label 3)')
plt.plot(np.linspace(J34ini, J34fin, number_iterations), np.transpose(energy_tracker_)[2,:], label ='third triplet list (label 4)')
'''
plt.plot(np.linspace(J34ini, J34fin, number_iterations), np.transpose(energy_tracker_)[0,:], label ='E1')
plt.plot(np.linspace(J34ini, J34fin, number_iterations), np.transpose(energy_tracker_)[1,:], label ='E2')
plt.plot(np.linspace(J34ini, J34fin, number_iterations), np.transpose(energy_tracker_)[2,:], label ='E3')
plt.legend()
plt.xlabel('$J_{23}$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
plt.title('Energies of the triplet subspace')
plt.show()


# finally we colour plot the expect_tracker matrix data array and the amplitude_tracker for the fourier transform
x = t
y = np.linspace(J23ini, J23fin, number_iterations)
plt.pcolormesh(x, y, expect_tracker)
plt.title('Oscillations in the triplet subspace')
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{23}$ ($E_0$)')
plt.colorbar()
plt.show()

# to plot the whole frequency spectrum
x = sample_freq
y = np.linspace(J23ini, J23fin, number_iterations)
plt.pcolormesh(x[:80], y, amplitude_tracker[:,:80])
plt.title('Fourier transform of the oscillations in the triplet subspace')
plt.xlabel('frequency ($f_0$)')
plt.ylabel('$J_{23}$ ($E_0$)')
plt.colorbar()
plt.show()

"""
# We plot the expectation value of the Gell-Mann matrices
x = t
y = np.linspace(J23ini, J23fin, number_iterations)
plt.pcolormesh(x, y, et1)
plt.title('Oscillations in the triplet subspace')
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{23}$ ($E_0$)')
plt.colorbar()
plt.show()

# we plot the expectation value of the Gell-mann matrices
x = t
y = np.linspace(J23ini, J23fin, number_iterations)
plt.pcolormesh(x, y, et2)
plt.title('Oscillations in the triplet subspace')
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{23}$ ($E_0$)')
plt.colorbar()
plt.show()

# we plot the expectation value of the Gell-mann matrices
x = t
y = np.linspace(J23ini, J23fin, number_iterations)
plt.pcolormesh(x, y, et3)
plt.title('Oscillations in the triplet subspace')
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{23}$ ($E_0$)')
plt.colorbar()
plt.show()

# we plot the expectation value of the Gell-mann matrices
x = t
y = np.linspace(J23ini, J23fin, number_iterations)
plt.pcolormesh(x, y, et4)
plt.title('Oscillations in the triplet subspace')
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{23}$ ($E_0$)')
plt.colorbar()
plt.show()

x = t
y = np.linspace(J23ini, J23fin, number_iterations)
plt.pcolormesh(x, y, et5)
plt.title('Oscillations in the triplet subspace')
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{23}$ ($E_0$)')
plt.colorbar()
plt.show()

# we plot the expectation value of the Gell-mann matrices
x = t
y = np.linspace(J23ini, J23fin, number_iterations)
plt.pcolormesh(x, y, et6)
plt.title('Oscillations in the triplet subspace')
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{23}$ ($E_0$)')
plt.colorbar()
plt.show()

# we plot the expectation value of the Gell-mann matrices
x = t
y = np.linspace(J23ini, J23fin, number_iterations)
plt.pcolormesh(x, y, et7)
plt.title('Oscillations in the triplet subspace')
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{23}$ ($E_0$)')
plt.colorbar()
plt.show()

# we plot the expectation value of the Gell-mann matrices
x = t
y = np.linspace(J23ini, J23fin, number_iterations)
plt.pcolormesh(x, y, et8)
plt.title('Oscillations in the triplet subspace')
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{23}$ ($E_0$)')
plt.colorbar()
plt.show()
"""

# now we want to store the value of the traces and plot them over time
trace_dm_tracker =  np.zeros((number_iterations, dimensions_expect_tracker))
trace_reduced_tracker = np.zeros((number_iterations, dimensions_expect_tracker))
expect_singlet_tracker = np.zeros((number_iterations, dimensions_expect_tracker))
singlet = (tensor(basis(2,0), basis(2,1)) - tensor(basis(2,1), basis(2,0))).unit()
singlet_dm = singlet*singlet.dag()
expect_triplet_tracker = np.zeros((number_iterations, dimensions_expect_tracker))
triplet = (tensor(basis(2,1), basis(2,1)))
triplet_dm = triplet*triplet.dag()
amplitude_tracker_singlet = np.zeros((number_iterations, dimensions_expect_tracker))
amplitude_tracker_triplet = np.zeros((number_iterations, dimensions_expect_tracker))
amplitude_tracker_trace = np.zeros((number_iterations, dimensions_expect_tracker))

for l in range(number_iterations):
    print('reconstructing dm...')
    for k in range(dimensions_expect_tracker):
        # we get the traces of the squares of the density matrices
        m = density_matrices[l,k,:,:]
        m_pad = np.pad(m,((2,11),(2,11)))
        m_pad = Qobj(m_pad)
        W = Qobj(trans_matrix).dag()
        finaldm = basis_transformation(m_pad, W)
        trace_dm_tracker[l,k] = finaldm.tr()
        # now we want to trace out the information of spins 3 and 4
        finaldm.dims =  [[2, 2, 2, 2], [2, 2, 2, 2]]
        dm12 = finaldm.ptrace([0, 1])
        trace_reduced_tracker[l,k] = (dm12*dm12).tr()
        
        # we get the expectation value of singlet in spins 1 and 2
        expect_singlet_tracker[l,k] = (dm12*singlet_dm).tr()
        expect_triplet_tracker[l,k] = (dm12*triplet_dm).tr()
        
    # fourier transform of the trace
    time_step = 1/sample_rate
    sig_fft_trace = sp.fftpack.fft(trace_reduced_tracker[l,:])
    amplitude_trace = np.abs(sig_fft_trace)
    sample_freq_trace = sp.fftpack.fftfreq(trace_reduced_tracker[l,:].size, d = time_step)
    amplitude_trace[0] = 0 # we set the zero frequency amplitude
    amplitude_tracker_trace[l,:] = amplitude_trace
    
    # fourier transform of the probabilities
    time_step = 1/sample_rate
    sig_fft_singlet = sp.fftpack.fft(expect_singlet_tracker[l,:]) # we get the fft of the signal
    sig_fft_triplet = sp.fftpack.fft(expect_triplet_tracker[l,:])
    amplitude_singlet = np.abs(sig_fft_singlet)
    amplitude_triplet = np.abs(sig_fft_triplet)
    sample_freq_singlet = sp.fftpack.fftfreq(expect_singlet_tracker[l,:].size, d = time_step)
    sample_freq_triplet = sp.fftpack.fftfreq(expect_triplet_tracker[l,:].size, d = time_step)
    amplitude_singlet[0] = 0 # we set the zero frequency amplitude
    amplitude_triplet[0] = 0
    amplitude_tracker_singlet[l,:] = amplitude_singlet
    amplitude_tracker_triplet[l,:] = amplitude_triplet

print('done')
# we plot
x = t
y = np.linspace(J23ini, J23fin, number_iterations)
#plt.pcolormesh(x, y, trace_dm_tracker)
plt.pcolormesh(x, y, trace_reduced_tracker)
plt.title('Oscillations of the trace of the reduced density matrix squared') #spins 1 and 2
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{23}$ ($E_0$)')
plt.colorbar()
plt.show()

# the fourier of the reduced osicllations (trace)
x = sample_freq_trace
y = np.linspace(J23ini, J23fin, number_iterations)
plt.pcolormesh(x[:80], y, amplitude_tracker_trace[:,:80])
plt.title('Fourier transform of the oscillations of the trace of the reduced density matrix squared')
plt.xlabel('frequency ($f_0$)')
plt.ylabel('$J_{23}$ ($E_0$)')
plt.colorbar()
plt.show()

x = t
y = np.linspace(J23ini, J23fin, number_iterations)
#plt.pcolormesh(x, y, trace_dm_tracker)
plt.pcolormesh(x, y, expect_singlet_tracker)
plt.title('Oscillations of the reduced system (proba singlet)') #spins 1 and 2
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{23}$ ($E_0$)')
plt.colorbar()
plt.show()

# the fourier of the reduced osicllations (singlet)
x = sample_freq_singlet
y = np.linspace(J23ini, J23fin, number_iterations)
plt.pcolormesh(x[:80], y, amplitude_tracker_singlet[:,:80])
plt.title('Fourier transform of the oscillations of the reduced singlet')
plt.xlabel('frequency ($f_0$)')
plt.ylabel('$J_{23}$ ($E_0$)')
plt.colorbar()
plt.show()

x = t
y = np.linspace(J23ini, J23fin, number_iterations)
#plt.pcolormesh(x, y, trace_dm_tracker)
plt.pcolormesh(x, y, expect_triplet_tracker)
plt.title('Oscillations of the reduced system (proba triplet)') #spins 1 and 2
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{23}$ ($E_0$)')
plt.colorbar()
plt.show()

# the fourier of the reduced osicllations (triplet)
x = sample_freq_triplet
y = np.linspace(J23ini, J23fin, number_iterations)
plt.pcolormesh(x[:80], y, amplitude_tracker_triplet[:,:80])
plt.title('Fourier transform of the oscillations of the reduced triplet')
plt.xlabel('frequency ($f_0$)')
plt.ylabel('$J_{23}$ ($E_0$)')
plt.colorbar()
plt.show()