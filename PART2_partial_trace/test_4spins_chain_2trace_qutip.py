#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 18:25:39 2023

@author: ceboncompte
"""

from hamiltonian import hheis_general, heisenberg_hamiltonian_4, is_unitary
from use import matrix_plot, basis_transformation
from basis_matrix import coupled_matrix_gen

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

"""
Readout projections that give oscillations
S
T- (11)
01 state, 10 state
T0 state 
combinations ++, --, +-
combinations -0, 0-, 0+
combinations -1, 1-, 1+
equal superp 01, 10, 11
superp S T-
sum of reading out T-, T0 and T+ (it does: oscillations + 0 + 0)
"""

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

# We generate the basis-transformation matrix and we transform the states
trans_matrix = coupled_matrix_gen(spins)
TM0_t = basis_transformation(TM0, trans_matrix)
TM1_t = basis_transformation(TM1, trans_matrix).unit()
TM2_t = basis_transformation(TM2, trans_matrix).unit()
sup = (TM1_t + TM2_t).unit()
print(np.array(TM0_t), np.array(TM1_t), np.array(TM2_t))
minus_0 = TM1_t.eliminate_states([0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
print(np.array(minus_0))





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

dimensions_expect_tracker = 1000 # number of time steps
states = np.zeros((number_iterations, dimensions_expect_tracker, 3, 1), dtype=complex) #triplet subspace

expect_singlet_tracker = np.zeros((number_iterations, dimensions_expect_tracker))
singlet = (tensor(basis(2,0), basis(2,1)) - tensor(basis(2,1), basis(2,0))).unit()
singlet_dm = singlet*singlet.dag()

expect_tripletm_tracker = np.zeros((number_iterations, dimensions_expect_tracker))
tripletm = (tensor(basis(2,1), basis(2,1))) #triplet minus
tripletm_dm = tripletm*tripletm.dag()

expect_triplet0_tracker = np.zeros((number_iterations, dimensions_expect_tracker))
triplet0 = (tensor(basis(2,0), basis(2,1)) + tensor(basis(2,1), basis(2,0))).unit()
triplet0_dm = triplet0*triplet0.dag()

expect_tripletp_tracker = np.zeros((number_iterations, dimensions_expect_tracker))
tripletp = (tensor(basis(2,0), basis(2,0)))
tripletp_dm = tripletp*tripletp.dag()

expect_basis01_tracker = np.zeros((number_iterations, dimensions_expect_tracker))
basis01 = (tensor(basis(2,0), basis(2,1)))
#basis01 = (tensor(basis(2,1), basis(2,0)))
#basis01 = (tensor(basis(2,0), basis(2,1)) + tensor(basis(2,1), basis(2,0))).unit()
#plus = (basis(2,0) + basis(2,1)).unit()
#minus = (basis(2,0) - basis(2,1)).unit()
#basis01 = (tensor(plus, basis(2,0))).unit()
#basis01 = (tensor(basis(2,0), basis(2,1)) + tensor(basis(2,1), basis(2,0)) + tensor(basis(2,1), basis(2,1))).unit()
#basis01 = (singlet + tensor(basis(2,1), basis(2,1))).unit()
basis01_dm = basis01*basis01.dag()

amplitude_tracker_singlet = np.zeros((number_iterations, dimensions_expect_tracker))

amplitude_tracker_tripletm = np.zeros((number_iterations, dimensions_expect_tracker))
amplitude_tracker_triplet0 = np.zeros((number_iterations, dimensions_expect_tracker))
amplitude_tracker_tripletp = np.zeros((number_iterations, dimensions_expect_tracker))

amplitude_tracker_basis01 = np.zeros((number_iterations, dimensions_expect_tracker))

J12zero = 2

# We generate the basis-transformation matrix
trans_matrix = coupled_matrix_gen(spins)

for J34 in values_J34:
    print("constructing plots... %", (n/number_iterations)*100)
    # We first want to create the arbitrary Hamiltonian
    Jij_vector = np.array([J12zero-J34, 0.6, J34])
    spins = len(Jij_vector) + 1
    H = hheis_general(Jij_vector, spins, B)
    
    # We basis transform the Hamiltonian matrix
    H_coup = basis_transformation(H, trans_matrix)
    
    # We take the first triplet and we get the energies of the states (end)
    H_triplet = H_coup[2:5, 2:5]
    Htriplet = Qobj(H_coup[2:5, 2:5]*2*np.pi) # we put units of omega
    
    # --------------------------------------------- OSCILLATIONS
    
    # We solve the Schrodinger equation (QuTiP)
    times = np.linspace(0, end_time, dimensions_expect_tracker)
    ket0 = minus_0
    result = mesolve(Htriplet, ket0, times, [], [])
    
    # We store the states in a big matrix (states)
    for k in range(len(result.states)):
        #print(result.states[k].norm()) # this preserves norm
        states[n,k,:,:] = np.array(result.states[k])
        # We pad the state with zeros outside the first triplet
        psi = np.pad(states[n,k,:,:], ((2,11), (0,0)))
        psi = Qobj(psi)
        #print(psi.norm()) # this preserves norm
        W = Qobj(trans_matrix).dag()
        finalpsi = basis_transformation(psi, W)
        #print(finalpsi.norm()) # this preserves norm
        #finaldm = finalpsi*finalpsi.dag()
        #print((finaldm*finaldm).norm()) # states are pure!
        # Wecompute trace out information of spins 3 and 4
        finalpsi.dims =  [[2, 2, 2, 2], [1, 1, 1, 1]]
        dm12 = finalpsi.ptrace([1, 2]) #always returns dm
        #print(dm12)
        #print(dm12.norm()) # this preserves norm
        #print((dm12*dm12).norm()) # not pure states in general (good)
        # We get the probabilities
        expect_singlet_tracker[n,k] = (dm12*singlet_dm).tr()
        
        expect_tripletm_tracker[n,k] = (dm12*tripletm_dm).tr()
        expect_triplet0_tracker[n,k] = (dm12*triplet0_dm).tr()
        expect_tripletp_tracker[n,k] = (dm12*tripletp_dm).tr()
        
        expect_basis01_tracker[n,k] = (dm12*basis01_dm).tr()
        
    # Fourier ttransform
    sample_rate = dimensions_expect_tracker/end_time
    time_step = 1/sample_rate
    sig_fft_singlet = sp.fftpack.fft(expect_singlet_tracker[n,:])
    
    sig_fft_tripletm = sp.fftpack.fft(expect_tripletm_tracker[n,:])
    sig_fft_triplet0 = sp.fftpack.fft(expect_triplet0_tracker[n,:])
    sig_fft_tripletp = sp.fftpack.fft(expect_tripletp_tracker[n,:])
    
    sig_fft_basis01 = sp.fftpack.fft(expect_basis01_tracker[n,:])
    
    amplitude_singlet = np.abs(sig_fft_singlet)
    
    amplitude_tripletm = np.abs(sig_fft_tripletm)
    amplitude_triplet0 = np.abs(sig_fft_triplet0)
    amplitude_tripletp = np.abs(sig_fft_tripletp)
    
    amplitude_basis01 = np.abs(sig_fft_basis01)
    
    sample_freq_singlet = sp.fftpack.fftfreq(expect_singlet_tracker[n,:].size, d = time_step)
    
    sample_freq_tripletm = sp.fftpack.fftfreq(expect_tripletm_tracker[n,:].size, d = time_step)
    sample_freq_triplet0 = sp.fftpack.fftfreq(expect_triplet0_tracker[n,:].size, d = time_step)
    sample_freq_tripletp = sp.fftpack.fftfreq(expect_tripletp_tracker[n,:].size, d = time_step)
    
    sample_freq_basis01 = sp.fftpack.fftfreq(expect_basis01_tracker[n,:].size, d = time_step)
    
    amplitude_singlet[0] = 0
    
    amplitude_tripletm[0] = 0
    amplitude_triplet0[0] = 0
    amplitude_tripletp[0] = 0
    
    amplitude_basis01[0] = 0
    
    amplitude_tracker_singlet[n,:] = amplitude_singlet
    
    amplitude_tracker_tripletm[n,:] = amplitude_tripletm
    amplitude_tracker_triplet0[n,:] = amplitude_triplet0
    amplitude_tracker_tripletp[n,:] = amplitude_tripletp
    
    amplitude_tracker_basis01[n,:] = amplitude_basis01
    
    # We diagonalize and obtain energy values
    ev = sp.linalg.eigvalsh(H_triplet)
    energy_tracker[n,:] = ev
    n+=1

n = 0

# we plot the energy of the eigenstates
plt.figure()
plt.plot(J12zero-2*values_J34, energy_tracker[:,0], label ='$E_1$')
plt.plot(J12zero-2*values_J34, energy_tracker[:,1], label ='$E_2$')
plt.plot(J12zero-2*values_J34, energy_tracker[:,2], label ='$E_3$')
plt.axvline(x=0, color='k', linestyle = 'dotted')
plt.legend()
plt.xlabel('$J_{12} - J_{34}$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
#plt.title('Energies of the triplet subspace')
plt.show()

# oscillations of the probability of singlet
x = times
y = J12zero-2*values_J34
#plt.pcolormesh(x, y, trace_dm_tracker)
plt.pcolormesh(x, y, expect_singlet_tracker)
#plt.title('Oscillations of the reduced system (proba singlet)') #spins 1 and 2
plt.axhline(y=0, color='w', linestyle = 'dotted')
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{12} - J_{34}$ ($E_0$)')
plt.colorbar()
plt.show()

# the fourier of the reduced osicllations (singlet)
x = sample_freq_singlet
y = J12zero-2*values_J34
plt.pcolormesh(x[:80], y, amplitude_tracker_singlet[:,:80])
#plt.title('Fourier transform of the oscillations of the reduced singlet')
plt.axhline(y=0, color='w', linestyle = 'dotted')
plt.xlabel('frequency ($f_0$)')
plt.ylabel('$J_{12} - J_{34}$ ($E_0$)')
plt.colorbar()
plt.show()

# Uncomment if we want probabilities of more states
"""
# oscillations of the probability of tripletm
x = times
y = np.linspace(J12zero-2*J34ini, J12zero-2*J34fin, number_iterations)
#plt.pcolormesh(x, y, trace_dm_tracker)
plt.pcolormesh(x, y, expect_tripletm_tracker)
#plt.title('Oscillations of the reduced system (proba triplet minus)') #spins 1 and 2
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{12} - J_{34}$ ($E_0$)')
plt.colorbar()
plt.show()

# the fourier of the reduced osicllations (tripletm)
x = sample_freq_tripletm
y = np.linspace(J12zero-2*J34ini, J12zero-2*J34fin, number_iterations)
plt.pcolormesh(x[:80], y, amplitude_tracker_tripletm[:,:80])
#plt.title('Fourier transform of the oscillations of the reduced triplet minus')
plt.xlabel('frequency ($f_0$)')
plt.ylabel('$J_{12} - J_{34}$ ($E_0$)')
plt.colorbar()
plt.show()

# oscillations of the probability of triplet0
x = times
y = np.linspace(J12zero-2*J34ini, J12zero-2*J34fin, number_iterations)
#plt.pcolormesh(x, y, trace_dm_tracker)
plt.pcolormesh(x, y, expect_triplet0_tracker)
#plt.title('Oscillations of the reduced system (proba triplet 0)') #spins 1 and 2
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{12} - J_{34}$ ($E_0$)')
plt.colorbar()
plt.show()

# the fourier of the reduced osicllations (triplet0)
x = sample_freq_triplet0
y = np.linspace(J12zero-2*J34ini, J12zero-2*J34fin, number_iterations)
plt.pcolormesh(x[:80], y, amplitude_tracker_triplet0[:,:80])
#plt.title('Fourier transform of the oscillations of the reduced triplet 0')
plt.xlabel('frequency ($f_0$)')
plt.ylabel('$J_{12} - J_{34}$ ($E_0$)')
plt.colorbar()
plt.show()

# oscillations of the probability of tripletp
x = times
y = np.linspace(J12zero-2*J34ini, J12zero-2*J34fin, number_iterations)
#plt.pcolormesh(x, y, trace_dm_tracker)
plt.pcolormesh(x, y, expect_tripletp_tracker)
#plt.title('Oscillations of the reduced system (proba triplet plus)') #spins 1 and 2
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{12} - J_{34}$ ($E_0$)')
plt.colorbar()
plt.show()

# the fourier of the reduced osicllations (tripletp)
x = sample_freq_tripletp
y = np.linspace(J12zero-2*J34ini, J12zero-2*J34fin, number_iterations)
plt.pcolormesh(x[:80], y, amplitude_tracker_tripletp[:,:80])
#plt.title('Fourier transform of the oscillations of the reduced triplet plus')
plt.xlabel('frequency ($f_0$)')
plt.ylabel('$J_{12} - J_{34}$ ($E_0$)')
plt.colorbar()
plt.show()

# oscillations of the probability of tripletm + triplet0 + tripletp
x = times
y = np.linspace(J12zero-2*J34ini, J12zero-2*J34fin, number_iterations)
#plt.pcolormesh(x, y, trace_dm_tracker)
plt.pcolormesh(x, y, expect_tripletm_tracker + expect_triplet0_tracker + expect_tripletp_tracker)
plt.title('Oscillations of the reduced system (proba triplet)') #spins 1 and 2
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{12} - J_{34}$ ($E_0$)')
plt.colorbar()
plt.show()

# oscillations of the probability of basis 01 state
x = times
y = np.linspace(J12zero-2*J34ini, J12zero-2*J34fin, number_iterations)
#plt.pcolormesh(x, y, trace_dm_tracker)
plt.pcolormesh(x, y, expect_basis01_tracker)
#plt.title('Oscillations of the reduced system (proba state 01)') #spins 1 and 2
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{12} - J_{34}$ ($E_0$)')
plt.colorbar()
plt.show()

# the fourier of the reduced osicllations (basis 01)
x = sample_freq_basis01
y = np.linspace(J12zero-2*J34ini, J12zero-2*J34fin, number_iterations)
plt.pcolormesh(x[:80], y, amplitude_tracker_basis01[:,:80])
#plt.title('Fourier transform of the oscillations of the reduced 01 state')
plt.xlabel('frequency ($f_0$)')
plt.ylabel('$J_{12} - J_{34}$ ($E_0$)')
plt.colorbar()
plt.show()
"""





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
states = np.zeros((number_iterations, dimensions_expect_tracker, 3, 1), dtype=complex) #triplet subspace

expect_singlet_tracker = np.zeros((number_iterations, dimensions_expect_tracker)) #singlet and triplet already defined

expect_tripletm_tracker = np.zeros((number_iterations, dimensions_expect_tracker))
expect_triplet0_tracker = np.zeros((number_iterations, dimensions_expect_tracker))
expect_tripletp_tracker = np.zeros((number_iterations, dimensions_expect_tracker))

expect_basis01_tracker = np.zeros((number_iterations, dimensions_expect_tracker))

amplitude_tracker_singlet = np.zeros((number_iterations, dimensions_expect_tracker))

amplitude_tracker_tripletm = np.zeros((number_iterations, dimensions_expect_tracker))
amplitude_tracker_triplet0 = np.zeros((number_iterations, dimensions_expect_tracker))
amplitude_tracker_tripletp = np.zeros((number_iterations, dimensions_expect_tracker))

amplitude_tracker_basis01 = np.zeros((number_iterations, dimensions_expect_tracker))

# We generate the basis-transformation matrix
trans_matrix = coupled_matrix_gen(spins)

for J23 in values_J23:
    print("constructing plots... %", (n/number_iterations)*100)
    # We first want to create the arbitrary Hamiltonian and print the matrix
    Jij_vector = np.array([2-1, J23, 1])
    spins = len(Jij_vector) + 1
    H = hheis_general(Jij_vector, spins, B)
    
    # We basis transform the Hamiltonian matrix
    H_coup = basis_transformation(H, trans_matrix)
    
    # We take the first triplet and we get the energies of the states (end)
    H_triplet = H_coup[2:5, 2:5]
    Htriplet = Qobj(H_coup[2:5, 2:5]*2*np.pi) # we put units of omega
    
    # --------------------------------------------- OSCILLATIONS
    
    # We solve the Schrodinger equation (QuTiP)
    times = np.linspace(0, end_time, dimensions_expect_tracker)
    ket0 = minus_0 #ket0 = basis(3, 1)
    result = mesolve(Htriplet, ket0, times, [], [])
    
    # We store the states in a big matrix (states)
    for k in range(len(result.states)):
        #print(result.states[k].norm()) # this preserves norm
        states[n,k,:,:] = np.array(result.states[k])
        # We pad the state with zeros outside the first triplet
        psi = np.pad(states[n,k,:,:], ((2,11), (0,0)))
        psi = Qobj(psi)
        #print(psi.norm()) # this preserves norm
        W = Qobj(trans_matrix).dag()
        finalpsi = basis_transformation(psi, W)
        #print(finalpsi.norm()) # this preserves norm
        #finaldm = finalpsi*finalpsi.dag()
        #print((finaldm*finaldm).norm()) # states are pure!
        # Wecompute trace out information of spins 3 and 4
        finalpsi.dims =  [[2, 2, 2, 2], [1, 1, 1, 1]]
        dm12 = finalpsi.ptrace([0, 1]) #always returns dm
        #print(dm12)
        #print(dm12.norm()) # this preserves norm
        #print((dm12*dm12).norm()) # not pure states in general (good)
        # We get the probabilities
        expect_singlet_tracker[n,k] = (dm12*singlet_dm).tr()
        
        expect_tripletm_tracker[n,k] = (dm12*tripletm_dm).tr()
        expect_triplet0_tracker[n,k] = (dm12*triplet0_dm).tr()
        expect_tripletp_tracker[n,k] = (dm12*tripletp_dm).tr()
        
        expect_basis01_tracker[n,k] = (dm12*basis01_dm).tr()
        
    # Fourier ttransform
    sample_rate = dimensions_expect_tracker/end_time
    time_step = 1/sample_rate
    sig_fft_singlet = sp.fftpack.fft(expect_singlet_tracker[n,:])
    
    sig_fft_tripletm = sp.fftpack.fft(expect_tripletm_tracker[n,:])
    sig_fft_triplet0 = sp.fftpack.fft(expect_triplet0_tracker[n,:])
    sig_fft_tripletp = sp.fftpack.fft(expect_tripletp_tracker[n,:])
    
    sig_fft_basis01 = sp.fftpack.fft(expect_basis01_tracker[n,:])
    
    amplitude_singlet = np.abs(sig_fft_singlet)
    
    amplitude_tripletm = np.abs(sig_fft_tripletm)
    amplitude_triplet0 = np.abs(sig_fft_triplet0)
    amplitude_tripletp = np.abs(sig_fft_tripletp)
    
    amplitude_basis01 = np.abs(sig_fft_basis01)
    
    sample_freq_singlet = sp.fftpack.fftfreq(expect_singlet_tracker[n,:].size, d = time_step)
    
    sample_freq_tripletm = sp.fftpack.fftfreq(expect_tripletm_tracker[n,:].size, d = time_step)
    sample_freq_triplet0 = sp.fftpack.fftfreq(expect_triplet0_tracker[n,:].size, d = time_step)
    sample_freq_tripletp = sp.fftpack.fftfreq(expect_tripletp_tracker[n,:].size, d = time_step)
    
    sample_freq_basis01 = sp.fftpack.fftfreq(expect_basis01_tracker[n,:].size, d = time_step)
    
    amplitude_singlet[0] = 0
    
    amplitude_tripletm[0] = 0
    amplitude_triplet0[0] = 0
    amplitude_tripletp[0] = 0
    
    amplitude_basis01[0] = 0
    
    amplitude_tracker_singlet[n,:] = amplitude_singlet
    
    amplitude_tracker_tripletm[n,:] = amplitude_tripletm
    amplitude_tracker_triplet0[n,:] = amplitude_triplet0
    amplitude_tracker_tripletp[n,:] = amplitude_tripletp
    
    amplitude_tracker_basis01[n,:] = amplitude_basis01
    
    # We diagonalize and obtain energy values
    ev = sp.linalg.eigvalsh(H_triplet)
    energy_tracker_[n,:] = ev
    n+=1

n = 0

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

# oscillations of the probability of singlet
x = times
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

# oscillations of the probability of tripletm
x = times
y = np.linspace(J23ini, J23fin, number_iterations)
#plt.pcolormesh(x, y, trace_dm_tracker)
plt.pcolormesh(x, y, expect_tripletm_tracker)
plt.title('Oscillations of the reduced system (proba triplet minus)') #spins 1 and 2
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{23}$ ($E_0$)')
plt.colorbar()
plt.show()

# the fourier of the reduced osicllations (tripletm)
x = sample_freq_tripletm
y = np.linspace(J23ini, J23fin, number_iterations)
plt.pcolormesh(x[:80], y, amplitude_tracker_tripletm[:,:80])
plt.title('Fourier transform of the oscillations of the reduced triplet minus')
plt.xlabel('frequency ($f_0$)')
plt.ylabel('$J_{23}$ ($E_0$)')
plt.colorbar()
plt.show()

# oscillations of the probability of triplet0
x = times
y = np.linspace(J23ini, J23fin, number_iterations)
#plt.pcolormesh(x, y, trace_dm_tracker)
plt.pcolormesh(x, y, expect_triplet0_tracker)
plt.title('Oscillations of the reduced system (proba triplet 0)') #spins 1 and 2
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{23}$ ($E_0$)')
plt.colorbar()
plt.show()

# the fourier of the reduced osicllations (triplet0)
x = sample_freq_triplet0
y = np.linspace(J23ini, J23fin, number_iterations)
plt.pcolormesh(x[:80], y, amplitude_tracker_triplet0[:,:80])
plt.title('Fourier transform of the oscillations of the reduced triplet 0')
plt.xlabel('frequency ($f_0$)')
plt.ylabel('$J_{23}$ ($E_0$)')
plt.colorbar()
plt.show()

# oscillations of the probability of tripletp
x = times
y = np.linspace(J23ini, J23fin, number_iterations)
#plt.pcolormesh(x, y, trace_dm_tracker)
plt.pcolormesh(x, y, expect_tripletp_tracker)
plt.title('Oscillations of the reduced system (proba triplet plus)') #spins 1 and 2
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{23}$ ($E_0$)')
plt.colorbar()
plt.show()

# the fourier of the reduced osicllations (tripletp)
x = sample_freq_tripletp
y = np.linspace(J23ini, J23fin, number_iterations)
plt.pcolormesh(x[:80], y, amplitude_tracker_tripletp[:,:80])
plt.title('Fourier transform of the oscillations of the reduced triplet plus')
plt.xlabel('frequency ($f_0$)')
plt.ylabel('$J_{23}$ ($E_0$)')
plt.colorbar()
plt.show()

# oscillations of the probability of triplet
x = times
y = np.linspace(J23ini, J23fin, number_iterations)
#plt.pcolormesh(x, y, trace_dm_tracker)
plt.pcolormesh(x, y, expect_tripletm_tracker + expect_triplet0_tracker + expect_tripletp_tracker)
plt.title('Oscillations of the reduced system (proba triplet)') #spins 1 and 2
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{23}$ ($E_0$)')
plt.colorbar()
plt.show()

# oscillations of the probability of basis 01 state
x = times
y = np.linspace(J12zero-2*J34ini, J12zero-2*J34fin, number_iterations)
#plt.pcolormesh(x, y, trace_dm_tracker)
plt.pcolormesh(x, y, expect_basis01_tracker)
plt.title('Oscillations of the reduced system (proba state 01)') #spins 1 and 2
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{12} - J_{34}$ ($E_0$)')
plt.colorbar()
plt.show()

# the fourier of the reduced osicllations (basis 01)
x = sample_freq_basis01
y = np.linspace(J12zero-2*J34ini, J12zero-2*J34fin, number_iterations)
plt.pcolormesh(x[:80], y, amplitude_tracker_basis01[:,:80])
plt.title('Fourier transform of the oscillations of the reduced 01 state')
plt.xlabel('frequency ($f_0$)')
plt.ylabel('$J_{12} - J_{34}$ ($E_0$)')
plt.colorbar()
plt.show()