#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 09:49:40 2023

@author: ceboncompte
"""

from hamiltonian import hheis_general, heisenberg_hamiltonian_4, ladder_exchanges, is_unitary
from use import matrix_plot, basis_transformation
from basis_matrix import coupled_matrix_gen

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

"""
number_iterations = 100
end_time = 30
dimensions_expect_tracker = 1000

total time: 31 min

We need a bit more precision on the oscillations
number_iterations = 100
end_time = 30
dimensions_expect_tracker = 2000

total time: 34 min

We need a bit more precision
number_iterations = 300
end_time = 30
dimensions_expect_tracker = 2000

total time: 1h 43 min
"""

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

# We propose
S13S24 = (tensor(Tplus, Tminus) + tensor(Tminus, Tplus) - tensor(T0, T0) + tensor(S, S)).unit()
#print(S13S24)
trans_matrix = coupled_matrix_gen(spins)
ketini = tensor(S, S, S, S)
ketini_t = basis_transformation(ketini, trans_matrix).unit()
ketini_t = basis(14,0) #same

#state = tensor(S, S, S, S)
state = tensor(S13S24, S, S)
#state = tensor(S, S, S13S24)
#state = tensor(S, S13S24, S)
#state = tensor(S13S24, S13S24)
print(state)
state_t = basis_transformation(state, trans_matrix).unit()
print(state_t)
#ket0 = basis(14,0) #is the same
ket0 = (basis(14,0) + np.sqrt(3)*basis(14,1)).unit() #is the same
#ket0 = (basis(14,0) + np.sqrt(3)*basis(14,13)).unit() #is the same
#ket0 = (basis(14,0) + np.sqrt(3)*basis(14,10)).unit() #is the same
#ket0 = (basis(14,0) + np.sqrt(3)*basis(14,1) + 3*basis(14,12) + np.sqrt(3)*basis(14,13)).unit() #is the same
print(ket0)





# HERE WE START WITH THE PROTOCOL

# We sweep antisimetrically J12 and J34

# --------------------------------------------- FOR THE ENERGY PLOTS
number_iterations = 100 # number of different values of J we test
energy_tracker = np.zeros((number_iterations, 14)) # septuplet refers to the number of states in the subspace
J46ini = 1.5
J46fin = 0.5
values_J46 = np.linspace(J46ini, J46fin, number_iterations)
#values_J46 = np.linspace(2, 0, number_iterations)
n = 0

# --------------------------------------------- FOR THE OSCILLATIONS AND FOURIERS
end_time = 30 # time of the oscillations (in natural units)

dimensions_expect_tracker = 2000 # number of time steps
states = np.zeros((number_iterations, dimensions_expect_tracker, 14, 1), dtype=complex) #tsinglets

expect_singlet_tracker = np.zeros((number_iterations, dimensions_expect_tracker))
singlet = (tensor(basis(2,0), basis(2,1)) - tensor(basis(2,1), basis(2,0))).unit()
singlet_dm = singlet*singlet.dag()
singlet_dm = ket0*ket0.dag()
#singlet_dm = state*state.dag()

expect_triplet_tracker = np.zeros((number_iterations, dimensions_expect_tracker))
triplet = (tensor(basis(2,1), basis(2,1))) #tiplet minus
triplet_dm = triplet*triplet.dag()

triplet_dm = singlet_dm #erase

amplitude_tracker_singlet = np.zeros((number_iterations, dimensions_expect_tracker))
amplitude_tracker_triplet = np.zeros((number_iterations, dimensions_expect_tracker))

J35zero = 2

for J46 in values_J46:
    print('constructing plots...', (n/len(values_J46))*100, "%")
    # We first want to create the arbitrary Hamiltonian
    Jij_vector = np.array([1, 0, 1, 0, 1, 0, 1])
    Jij_ladder = np.array([1, 1, J35zero - J46, J46, 1, 1])
    spins = len(Jij_vector) + 1
    H = hheis_general(Jij_vector, spins, B) + ladder_exchanges(Jij_ladder, spins)
    
    # We generate the basis-transformation matrix
    trans_matrix = coupled_matrix_gen(spins)
    
    # We basis transform the Hamiltonian matrix
    H_coup = basis_transformation(H, trans_matrix)
    
    # We take the first triplet and we get the energies of the states (end)
    H_triplet = H_coup[0:14, 0:14]
    H_sept = Qobj(H_coup[0:14, 0:14]*2*np.pi)
    
    # --------------------------------------------- OSCILLATIONS
    
    # We solve the Schrodinger equation (QuTiP)
    times = np.linspace(0, end_time, dimensions_expect_tracker)
    #ket0 = minus_0 #already defined
    result = mesolve(H_sept, ketini_t, times, [], [])
    
    # We store the states in a big matrix (states)
    for k in range(len(result.states)):
        #print(result.states[k].norm()) # this preserves norm
        states[n,k,:,:] = np.array(result.states[k])
        
        #Uncomment if we want to partial trace
        """
        # We pad the state with zeros outside the first triplet
        psi = np.pad(states[n,k,:,:], ((198,51), (0,0))
        psi = Qobj(psi)
        #print(psi.norm()) # this preserves norm
        W = Qobj(trans_matrix).dag()
        finalpsi = basis_transformation(psi, W)
        #print(finalpsi.norm()) # this preserves norm
        #finaldm = finalpsi*finalpsi.dag()
        #print((finaldm*finaldm).norm()) # states are pure!
        # Wecompute trace out information of spins 3 and 4
        finalpsi.dims =  [[2, 2, 2, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1, 1, 1]]
        dm12 = finalpsi.ptrace([0, 1]) #always returns dm
        #print(dm12)
        #print(dm12.norm()) # this preserves norm
        #print((dm12*dm12).norm()) # not pure states in general (good)
        
        # We get the probabilities
        expect_singlet_tracker[n,k] = (dm12*singlet_dm).tr() # if complex to float error make it real
        expect_triplet_tracker[n,k] = (dm12*triplet_dm).tr()
        """
        
        # Uncomment if we dont want partial trace
        psi = Qobj(states[n,k,:,:])
        dm12 = psi*psi.dag()
        expect_singlet_tracker[n,k] = np.real((dm12*singlet_dm).tr())
        expect_triplet_tracker[n,k] = np.real((dm12*triplet_dm).tr())

    # Fourier ttransform
    sample_rate = dimensions_expect_tracker/end_time
    time_step = 1/sample_rate
    sig_fft_singlet = sp.fftpack.fft(expect_singlet_tracker[n,:])
    sig_fft_triplet = sp.fftpack.fft(expect_triplet_tracker[n,:])
    amplitude_singlet = np.abs(sig_fft_singlet)
    amplitude_triplet = np.abs(sig_fft_triplet)
    sample_freq_singlet = sp.fftpack.fftfreq(expect_singlet_tracker[n,:].size, d = time_step)
    sample_freq_triplet = sp.fftpack.fftfreq(expect_triplet_tracker[n,:].size, d = time_step)
    amplitude_singlet[0] = 0
    amplitude_triplet[0] = 0
    amplitude_tracker_singlet[n,:] = amplitude_singlet
    amplitude_tracker_triplet[n,:] = amplitude_triplet
    
    # We diagonalize and obtain energy values
    ev = sp.linalg.eigvalsh(H_triplet)
    energy_tracker[n,:] = ev
    n+=1

# we plot the energy of the eigenstates
plt.figure()
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[0,:], label ='$E1$')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[1,:], label ='$E2$')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[2,:], label ='$E3$')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[3,:], label ='$E4$')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[4,:], label ='$E5$')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[5,:], label ='$E6$')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[6,:], label ='$E7$')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[7,:], label ='$E8$')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[8,:], label ='$E9$')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[9,:], label ='$E10$')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[10,:], label ='$E11$')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[11,:], label ='$E12$')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[12,:], label ='$E13$')
plt.plot(J35zero-2*np.linspace(J46ini, J46fin, number_iterations), np.transpose(energy_tracker)[13,:], label ='$E14$')
plt.legend()
plt.xlabel('$J_{35} - J_{46}$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
plt.title('Energies of the singlet subspace')
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
plt.title('Energy differences of adjacent states of the singlet subspace')
plt.show()

# oscillations of the probability of singlet
x = times
y = np.linspace(J35zero-2*J46ini, J35zero-2*J46fin, number_iterations)
#plt.pcolormesh(x, y, trace_dm_tracker)
plt.pcolormesh(x, y, expect_singlet_tracker)
plt.title('Oscillations of the system in configuration 2')
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{35} - J_{46}$ ($E_0$)')
plt.colorbar()
plt.show()

# the fourier of the reduced osicllations (singlet)
x = sample_freq_singlet
y = np.linspace(J35zero-2*J46ini, J35zero-2*J46fin, number_iterations)
plt.pcolormesh(x[:50], y, amplitude_tracker_singlet[:,:50])
plt.title('Fourier transform of the oscillations')
plt.xlabel('frequency ($f_0$)')
plt.ylabel('$J_{35} - J_{46}$ ($E_0$)')
plt.colorbar()
plt.show()

# oscillations of the probability of triplet
x = times
y = np.linspace(J35zero-2*J46ini, J35zero-2*J46fin, number_iterations)
#plt.pcolormesh(x, y, trace_dm_tracker)
plt.pcolormesh(x, y, expect_triplet_tracker)
plt.title('Oscillations of the reduced system (proba triplet)') #spins 1 and 2
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{35} - J_{46}$ ($E_0$)')
plt.colorbar()
plt.show()

# the fourier of the reduced osicllations (triplet)
x = sample_freq_triplet
y = np.linspace(J35zero-2*J46ini, J35zero-2*J46fin, number_iterations)
plt.pcolormesh(x[:50], y, amplitude_tracker_triplet[:,:50])
plt.title('Fourier transform of the oscillations of the reduced triplet')
plt.xlabel('frequency ($f_0$)')
plt.ylabel('$J_{35} - J_{46}$ ($E_0$)')
plt.colorbar()
plt.show()