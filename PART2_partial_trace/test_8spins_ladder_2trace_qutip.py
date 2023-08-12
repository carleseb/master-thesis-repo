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

# For the initial state we examine the pairs basis
S = (basis(4,1) - basis(4,2)).unit()
T0 = (basis(4,1) + basis(4,2)).unit()
Tplus = basis(4,0)
Tminus = basis(4,3)

S13T24 = (basis(16,7) - basis(16,13)).unit()
T13S24 = (basis(16,11) - basis(16,14)).unit()

spins = 8
B = 0.5
trans_matrix = coupled_matrix_gen(spins)

# Intitial state
ket0 = tensor(S13T24, Tminus, Tminus)
ket0_t = basis_transformation(ket0, trans_matrix)
ket0_red = (basis(7,2) + np.sqrt(3)*basis(7,6)).unit() #same as ket0_t
dm0 = ket0_red * ket0_red.dag()
dm0 = np.array(dm0)

ket1 = tensor(Tminus, S13T24, Tminus)
ket1_t = basis_transformation(ket1, trans_matrix)
ket1_red = (basis(7,0) + np.sqrt(15)*basis(7,1) - np.sqrt(8)*basis(7,6)).unit()
dm1 = ket1_red * ket1_red.dag()
dm1 = np.array(dm1)

ket2 = tensor(Tminus, Tminus, S13T24)
ket2_t = basis_transformation(ket2, trans_matrix)
ket2_red = (-np.sqrt(24)*basis(7,1) + basis(7,4) + np.sqrt(35)*basis(7,5)).unit()
dm2 = ket2_red * ket2_red.dag()
dm2 = np.array(dm2)

ket3 = tensor(S, Tminus, Tminus, Tminus)
ket3_t = basis_transformation(ket3, trans_matrix)
ket3_red = basis(7,3)
dm3 = ket3_red * ket3_red.dag()
dm3 = np.array(dm3)

ket4 = tensor(Tminus, S, Tminus, Tminus)
ket4_t = basis_transformation(ket4, trans_matrix)
ket4_red = (np.sqrt(2)*basis(7,0) - basis(7,6)).unit()
dm4 = ket4_red * ket4_red.dag()
dm4 = np.array(dm4)

ket5 = tensor(Tminus, Tminus, S, Tminus)
ket5_t = basis_transformation(ket5, trans_matrix)
ket5_red = (-np.sqrt(2)*basis(7,1) + np.sqrt(3)*basis(7,4)).unit()
dm5 = ket5_red * ket5_red.dag()
dm5 = np.array(dm5)

ket6 = tensor(Tminus, Tminus, Tminus, S)
ket6_t = basis_transformation(ket6, trans_matrix)
ket6_red = (np.sqrt(20)*basis(7,3) - np.sqrt(15)*basis(7,5)).unit()
dm6 = ket6_red * ket6_red.dag()
dm6 = np.array(dm6)

ket7 = tensor(T13S24, Tminus, Tminus)
ket7_t = basis_transformation(ket7, trans_matrix)
ket7_red = (np.sqrt(8)*basis(7,0) - np.sqrt(3)*basis(7,2) + basis(7,6)).unit()
dm7 = ket7_red * ket7_red.dag()
dm7 = np.array(dm7)

ket8 = tensor(Tminus, T13S24, Tminus)
ket8_t = basis_transformation(ket8, trans_matrix)
ket8_red = (-np.sqrt(15)*basis(7,0) + basis(7,1) + np.sqrt(24)*basis(7,4)).unit()
dm8 = ket8_red * ket8_red.dag()
dm8 = np.array(dm8)

ket9 = tensor(Tminus, Tminus, T13S24)
ket9_t = basis_transformation(ket9, trans_matrix)
ket9_red = (np.sqrt(48)*basis(7,3) - np.sqrt(35)*basis(7,4) + basis(7,5)).unit()
dm9 = ket9_red * ket9_red.dag()
dm9 = np.array(dm9)





# HERE WE START WITH THE PROTOCOL

# We sweep antisimetrically J12 and J34

# --------------------------------------------- FOR THE ENERGY PLOTS
number_iterations = 100 # number of different values of J we test
energy_tracker = np.zeros((number_iterations, 7)) # septuplet refers to the number of states in the subspace
J46ini = 1.5
J46fin = 0.5
values_J46 = np.linspace(J46ini, J46fin, number_iterations)
#values_J46 = np.linspace(2, 0, number_iterations)
n = 0

# Number of spins for this simulation and magnetic field
spins = 8
B = 0.5

# --------------------------------------------- FOR THE OSCILLATIONS AND FOURIERS
end_time = 40 # time of the oscillations

dimensions_expect_tracker = 100 # number of time steps
states = np.zeros((number_iterations, dimensions_expect_tracker, 7, 1), dtype=complex) #triplet subspace

expect_singlet_tracker = np.zeros((number_iterations, dimensions_expect_tracker))
singlet = (tensor(basis(2,0), basis(2,1)) - tensor(basis(2,1), basis(2,0))).unit()
singlet_dm = singlet*singlet.dag()
expect_triplet_tracker = np.zeros((number_iterations, dimensions_expect_tracker))
triplet = (tensor(basis(2,1), basis(2,1))) #tiplet minus
triplet_dm = triplet*triplet.dag()

amplitude_tracker_singlet = np.zeros((number_iterations, dimensions_expect_tracker))
amplitude_tracker_triplet = np.zeros((number_iterations, dimensions_expect_tracker))

J35zero = 2

# We generate the basis-transformation matrix
trans_matrix = coupled_matrix_gen(spins)

for J46 in values_J46:
    print('constructing plots...', (n/len(values_J46))*100, "%")
    # We first want to create the arbitrary Hamiltonian
    # ------------------------------ PROTOCOL A -----------------------------------
    """
    Jij_vector = np.array([1, 0, 1, 0, 1, 0, 1])
    Jij_ladder = np.array([1, 1, 2*(J35zero - J46), 2*J46, 1, 1])
    
    Jij_vector = np.array([J46, 0, J46, 0, J46, 0, J46])
    Jij_ladder = np.array([J46, J46, J35zero - J46, J35zero - J46, J46, J46])
    
    Jij_vector = np.array([1, 0, 1, 0, 1, 0, 1])
    Jij_ladder = np.array([1, 1, J46, 0, 1, 1])
    
    Jij_vector = np.array([J35zero - J46, 0, J35zero - J46, 0, J46, 0, J46])
    Jij_ladder = np.array([J35zero - J46, J35zero - J46, 0.2, 0.2, J46, J46])
    
    Jij_vector = np.array([1, 0, 1, 0, 1, 0, 1])
    Jij_ladder = np.array([1, 1, J46, J46, 1, 1])
    """
    # ------------------------------ PROTOCOL B -----------------------------------
    """
    Jij_vector = np.array([0.2, 0, 1.5, 0, 1.5, 0, 0.2])
    Jij_ladder = np.array([J35zero - J46, J46, J35zero - J46 , J46, J35zero - J46, J46])
    
    Jij_vector = np.array([0.4, 0, 0.3, 0, 0.8, 0, 0.75])
    Jij_ladder = np.array([J35zero - J46, J46, J35zero - J46 , J46, J35zero - J46, J46])
    
    Jij_vector = np.array([0, 0, J46, 0, J35zero - J46, 0, 0])
    Jij_ladder = np.array([1, 1, 1, 1, 1, 1])
    
    Jij_vector = np.array([0.1, 0, 1.5*J46, 0, 1.5*(J35zero - J46), 0, 0.3])
    Jij_ladder = np.array([1, 1, 1, 1, 1, 1])
    
    Jij_vector = np.array([2*J46, 0, 2*(J35zero - J46), 0, 2*(J35zero - J46), 0, 2*J46])
    Jij_ladder = np.array([1, 1, 1, 1, 1, 1])
    
    Jij_vector = np.array([0.5*J46, 0, 0.5*(J35zero - J46), 0, 0.5*(J35zero - J46), 0, 0.5*J46])
    Jij_ladder = np.array([1, 1, 1, 1, 1, 1])
    
    Jij_vector = np.array([J35zero - J46, 0, J35zero - J46, 0, J35zero - J46, 0, J35zero - J46])
    Jij_ladder = np.array([J46, J46, J46, J46, J46, J46])
    """    
    Jij_vector = np.array([J46, 0, J46, 0, J46, 0, J46])
    Jij_ladder = np.array([J46, J46, J35zero - J46, J35zero - J46, J46, J46])
    spins = len(Jij_vector) + 1
    H = hheis_general(Jij_vector, spins, B) + ladder_exchanges(Jij_ladder, spins)
    
    # We basis transform the Hamiltonian matrix
    H_coup = basis_transformation(H, trans_matrix)
    
    # We take the first triplet and we get the energies of the states (end)
    H_sept = H_coup[198:205, 198:205]
    Hsept = Qobj(H_coup[198:205, 198:205]*2*np.pi)
    
    # --------------------------------------------- OSCILLATIONS
    
    # We solve the Schrodinger equation (QuTiP)
    times = np.linspace(0, end_time, dimensions_expect_tracker)
    #ket0 = minus_0 #already defined
    result = mesolve(Hsept, ket6_red, times, [], [])
    
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
        # We trace out information of spins 3 and 4
        finalpsi.dims =  [[2, 2, 2, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1, 1, 1]]
        dm12 = finalpsi.ptrace([6, 7]) #always returns dm, spins 3 and 5 are indices 2 and 4
        #print(dm12)
        #print(dm12.norm()) # this preserves norm
        #print((dm12*dm12).norm()) # not pure states in general (good)
        # We get the probabilities
        expect_singlet_tracker[n,k] = (dm12*singlet_dm).tr()
        expect_triplet_tracker[n,k] = (dm12*triplet_dm).tr()
        
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
    
    # We diagonalize and obtain energy values (can be commented)
    ev = sp.linalg.eigvalsh(H_sept)
    energy_tracker[n,:] = ev
    n+=1
    
n = 0

# we plot the energy of the eigenstates
y = J35zero-2*values_J46
#y = 2*(J35zero-2*values_J46)
"""
plt.figure()
plt.plot(y, energy_tracker[:,0], label ='$E_1$')
plt.plot(y, energy_tracker[:,1], label ='$E_2$')
plt.plot(y, energy_tracker[:,2], label ='$E_3$')
plt.plot(y, energy_tracker[:,3], label ='$E_4$')
plt.plot(y, energy_tracker[:,4], label ='$E_5$')
plt.plot(y, energy_tracker[:,5], label ='$E_6$')
plt.plot(y, energy_tracker[:,6], label ='$E_7$')
plt.legend()
plt.xlabel('$J_{35} - J_{46}$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
#plt.title('Energies of the septuplet subspace')
plt.show()

# we plot the differences of succesive eigenstate's energies
plt.figure()
plt.plot(y, energy_tracker[:,1] - energy_tracker[:,0], label ='E_2 - E_1')
plt.plot(y, energy_tracker[:,2] - energy_tracker[:,1], label ='E_3 - E_2')
plt.plot(y, energy_tracker[:,3] - energy_tracker[:,2], label ='E_4 - E_3')
plt.plot(y, energy_tracker[:,4] - energy_tracker[:,3], label ='E_5 - E_4')
plt.plot(y, energy_tracker[:,5] - energy_tracker[:,4], label ='E_6 - E_5')
plt.plot(y, energy_tracker[:,6] - energy_tracker[:,5], label ='E_7 - E_6')
plt.legend()
plt.xlabel('$J_{35} - J_{46}$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
#plt.title('Energy differences of adjacent states of the septuplet subspace')
plt.show()
"""

# oscillations of the probability of singlet
x = times
plt.pcolormesh(x, y, expect_singlet_tracker)
plt.axhline(y=0, color='w', linestyle='dotted')
#plt.title('Oscillations of the reduced system (proba singlet)') #spins 1 and 2
plt.xlabel('time ($t_0$)')
#plt.ylabel('$J_{p1} - J_{p2}$ ($E_0$)') #for step 7A
#plt.ylabel('$J_{35} - J_{46}$ ($E_0$)') #for step 8A
plt.ylabel('$J_{mid} - J_p$ ($E_0$)') #for step 9A
plt.colorbar()
plt.show()

# the fourier of the reduced osicllations (singlet)
x = sample_freq_singlet
plt.pcolormesh(x[:50], y, amplitude_tracker_singlet[:,:50])
#plt.title('Fourier transform of the oscillations of the reduced singlet')
plt.axhline(y=0, color='w', linestyle='dotted')
plt.xlabel('frequency ($f_0$)')
#plt.ylabel('$J_{p1} - J_{p2}$ ($E_0$)') #for step 7A
#plt.ylabel('$J_{35} - J_{46}$ ($E_0$)') #for step 8A
plt.ylabel('$J_{mid} - J_p$ ($E_0$)') #for step 9A
plt.colorbar()
plt.show()

"""
# oscillations of the probability of triplet
x = times
y = np.linspace(J35zero-2*J46ini, J35zero-2*J46fin, number_iterations)
#plt.pcolormesh(x, y, trace_dm_tracker)
plt.pcolormesh(x, y, expect_triplet_tracker)
#plt.title('Oscillations of the reduced system (proba triplet)') #spins 1 and 2
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{35} - J_{46}$ ($E_0$)')
plt.colorbar()
plt.show()

# the fourier of the reduced osicllations (triplet)
x = sample_freq_triplet
y = np.linspace(J35zero-2*J46ini, J35zero-2*J46fin, number_iterations)
plt.pcolormesh(x[:50], y, amplitude_tracker_triplet[:,:50])
#plt.title('Fourier transform of the oscillations of the reduced triplet')
plt.xlabel('frequency ($f_0$)')
plt.ylabel('$J_{35} - J_{46}$ ($E_0$)')
plt.colorbar()
plt.show()
"""
