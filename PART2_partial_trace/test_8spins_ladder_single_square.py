#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 10:05:31 2023

@author: ceboncompte
"""

from hamiltonian import hheis_general, heisenberg_hamiltonian_4, ladder_exchanges, is_unitary
from use import matrix_plot, basis_transformation
from basis_matrix import coupled_matrix_gen

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# Number of spins for this simulation and magnetic field
spins = 8
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
S13T24 = (basis(16,7) - basis(16,13)).unit()

# We propose
trans_matrix = coupled_matrix_gen(spins)
state = tensor(S, Tminus, Tminus, Tminus)
state_t = basis_transformation(state, trans_matrix).unit()
print(state_t) # all zeros except one 1 at position 200
# this state belongs to the first setuplet [198:205] (205 not included)
ket0 = basis(7,2) # is the same

#Other
state_ = tensor(S13T24, Tminus, Tminus)
state_t_ = basis_transformation(state_, trans_matrix).unit()
print(state_t_)# 0.5 at 200, 0.866 at 204
ket1 = (basis(7,2) + np.sqrt(3)*basis(7,6)).unit()

# To know which eigenstates does the initial state ovelap with the most
Jij_vector = np.array([1, 0, 1, 0, 1, 0, 1])
Jij_ladder = np.array([1, 1, 0, 0, 1, 1])
spins = len(Jij_vector) + 1
H = hheis_general(Jij_vector, spins, B) + ladder_exchanges(Jij_ladder, spins)
H_t = basis_transformation(H, trans_matrix)
H_sep = Qobj(H_t[198:205,198:205])
es, eigs = H_sep.eigenstates()
for index, ei in enumerate(eigs):
    print('eigenstate ', index)
    print(ket1.overlap(ei))


# HERE WE START WITH THE PROTOCOL

# In this case we attempt to check if a tuning-square protocols works with a full ladder of exchanges
# It apears that if the mid couplings are low enough, like 2 orders of magnitude low it works

# --------------------------------------------- FOR THE ENERGY PLOTS
number_iterations = 100 # number of different values of J we test
energy_tracker = np.zeros((number_iterations, 7)) # septuplet refers to the number of states in the subspace
J46ini = 1.5
J46fin = 0.5
values_J46 = np.linspace(J46ini, J46fin, number_iterations)
#values_J46 = np.linspace(2, 0, number_iterations)
n = 0

# --------------------------------------------- FOR THE OSCILLATIONS AND FOURIERS
end_time = 30 # time of the oscillations (in natural units)

dimensions_expect_tracker = 1000 # number of time steps
states = np.zeros((number_iterations, dimensions_expect_tracker, 7, 1), dtype=complex) #septuplet subspace

#singlet is actually Tminus,S (ket0)
expect_singlet_tracker = np.zeros((number_iterations, dimensions_expect_tracker))
singlet_dm = TM1*TM1.dag()
singlet_dm.dims =  [[2, 2, 2, 2], [2, 2, 2, 2]]
amplitude_tracker_singlet = np.zeros((number_iterations, dimensions_expect_tracker))

#singlet_ is actually ket1
singlet_dm_ = S13T24*S13T24.dag()
singlet_dm_.dims =  [[2, 2, 2, 2], [2, 2, 2, 2]]
amplitude_tracker_singlet = np.zeros((number_iterations, dimensions_expect_tracker))

J35zero = 2

# We generate the basis-transformation matrix
trans_matrix = coupled_matrix_gen(spins)

for J46 in values_J46:
    print('constructing plots...', (n/len(values_J46))*100, "%")
    # We first want to create the arbitrary Hamiltonian
    #Jij_vector = np.array([J35zero - J46, 0, J46, 0, 1, 0, 1])
    #Jij_ladder = np.array([1.2, 0.7, 0, 0, 1, 1])
    
    #Jij_vector = np.array([0.6, 0, 0.6, 0, 1, 0, 1])
    #Jij_ladder = np.array([J35zero - J46, J46, 0, 0., 1, 1])
    
    Jij_vector = np.array([J46, 0, J46, 0, 1, 0, 1])
    Jij_ladder = np.array([J35zero - J46, J35zero - J46, 0, 0., 1, 1])
    spins = len(Jij_vector) + 1
    H = hheis_general(Jij_vector, spins, B) + ladder_exchanges(Jij_ladder, spins)
    
    # We basis transform the Hamiltonian matrix
    H_coup = basis_transformation(H, trans_matrix)
    
    # We take the first septuplet and we get the energies of the states (end)
    H_sept = H_coup[198:205, 198:205]
    Hsept = Qobj(H_coup[198:205, 198:205]*2*np.pi)
    
    # --------------------------------------------- OSCILLATIONS
    
    # We solve the Schrodinger equation (QuTiP)
    times = np.linspace(0, end_time, dimensions_expect_tracker)
    #ket0 = minus_0 #already defined
    result = mesolve(Hsept, ket1, times, [], []) #initial state
    
    # We store the states in a big matrix (states)
    for k in range(len(result.states)):
        #print(result.states[k].norm()) # this preserves norm
        states[n,k,:,:] = np.array(result.states[k])
        # We pad the state with zeros outside the first triplet
        psi = np.pad(states[n,k,:,:], ((198,51), (0,0)))
        psi = Qobj(psi)
        #print(psi.norm()) # this preserves norm
        W = Qobj(trans_matrix).dag()
        finalpsi = basis_transformation(psi, W)
        #print(finalpsi.norm()) # this preserves norm
        #finaldm = finalpsi*finalpsi.dag()
        #print((finaldm*finaldm).norm()) # states are pure!
        # We trace out information of spins 5, 6, 7, 8
        finalpsi.dims =  [[2, 2, 2, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1, 1, 1]]
        dm12 = finalpsi.ptrace([0, 1, 2, 3]) #always returns dm
        #print(dm12.dims)
        #print(dm12)
        #print(dm12.norm()) # this preserves norm
        #print((dm12*dm12).norm()) # not pure states in general (good)
        # We get the probabilities
        #expect_singlet_tracker[n,k] = (dm12*singlet_dm).tr() #careful state we measure
        expect_singlet_tracker[n,k] = np.real((dm12*singlet_dm_).tr())
        
    # Fourier ttransform
    sample_rate = dimensions_expect_tracker/end_time
    time_step = 1/sample_rate
    sig_fft_singlet = sp.fftpack.fft(expect_singlet_tracker[n,:])
    amplitude_singlet = np.abs(sig_fft_singlet)
    sample_freq_singlet = sp.fftpack.fftfreq(expect_singlet_tracker[n,:].size, d = time_step)
    amplitude_singlet[0] = 0
    amplitude_tracker_singlet[n,:] = amplitude_singlet
    
    
    # We diagonalize and obtain energy values
    ev = sp.linalg.eigvalsh(H_sept)
    energy_tracker[n,:] = ev
    
    n+=1

n = 0


# we plot the energy of the eigenstates
plt.figure()
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[0,:], label ='$E1$')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[1,:], label ='$E2$')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[2,:], label ='$E3$')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[3,:], label ='$E4$')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[4,:], label ='$E5$')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[5,:], label ='$E6$')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[6,:], label ='$E7$')
plt.legend()
plt.xlabel('$J_{35} - J_{46}$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
plt.title('Energies of the septuplet subspace')
plt.show()

# oscillations of the probability of singlet
x = times
y = np.linspace(J35zero-2*J46ini, J35zero-2*J46fin, number_iterations)
#plt.pcolormesh(x, y, trace_dm_tracker)
plt.pcolormesh(x, y, expect_singlet_tracker)
plt.title('Oscillations of the reduced system (proba singlet)') #spins 1 and 2
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{35} - J_{46}$ ($E_0$)')
plt.colorbar()
plt.show()

# the fourier of the reduced osicllations (singlet)
x = sample_freq_singlet
y = np.linspace(J35zero-2*J46ini, J35zero-2*J46fin, number_iterations)
plt.pcolormesh(x[:50], y, amplitude_tracker_singlet[:,:50])
plt.title('Fourier transform of the oscillations of the reduced singlet')
plt.xlabel('frequency ($f_0$)')
plt.ylabel('$J_{35} - J_{46}$ ($E_0$)')
plt.colorbar()
plt.show()