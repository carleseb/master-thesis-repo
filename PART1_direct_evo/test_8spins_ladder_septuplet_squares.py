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

S13T24 = (basis(16,7) - basis(16,13)).unit()
T13S24 = (basis(16,11) - basis(16,14)).unit()

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





# --------------------------------------------- FOR THE ENERGY PLOTS
number_iterations = 100 # number of different values of J we test
energy_tracker = np.zeros((number_iterations, 7)) # 7 (septuplet) refers to the number of states in the subspace
J46ini = 1.5
J46fin = 0.5
#J46ini = 1.25
#J46fin = 0.75
values_J46 = np.linspace(J46ini, J46fin, number_iterations)
n = 0

# --------------------------------------------- FOR THE OSCILLATIONS AND FOURIERS
end_time = 20 # time of the oscillations (in natural units)
dimensions_expect_tracker = 100 # number of time steps
expect_tracker = np.zeros((number_iterations, dimensions_expect_tracker))
amplitude_tracker = np.zeros((number_iterations, dimensions_expect_tracker))
J35zero = 2

# We generate the basis-transformation matrix
trans_matrix = coupled_matrix_gen(spins)

for J46 in values_J46:
    print('Obtaining plots... %', n*100/number_iterations)
    # We first want to create the arbitrary Hamiltonian and print the matrix
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
    
    Jij_vector = np.array([2*(J35zero - J46), 0, 1.5, 0, 1.5, 0, 2*J46])
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
    
    H = hheis_general(Jij_vector, spins, B) + ladder_exchanges(Jij_ladder, spins)
    
    # We plot the matrix
#    matrix_plot(H)
    
    # We finally basis transform and plot again the Hamiltonian matrix
    H_coup = basis_transformation(H, trans_matrix)
#    matrix_plot(H_coup)
    
    # We take the first septuplet and we get the energies of the states
    H_q = H_coup[198:205, 198:205]
    
    # --------------------------------------------- OSCILLATIONS
    # We transform this Hamiltonian into an H_channel (non time dependent)
    CH0 = H_channel(H_q)
    CH0.pulse.add_constant(2*np.pi*1.)
    
    # We solve the Schrodinger equation
    calculation = H_solver()
    calculation.add_channels(CH0)
    
    # calculate for end_time
    number_steps = dimensions_expect_tracker
    sample_rate = number_steps/end_time
    calculation.calculate(dm2, end_time = end_time, sample_rate = sample_rate)
    
    # calculate the expectatoin value of the matrix dm0 and plot
    # plus fourier transform
    dm0_expect = calculation.return_expectation_values(dm2)
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
"""
# ---------------------------------- PROTOCOL A ---------------------------------------

# we plot the energy of the eigenstates
plt.figure()
y = J35zero-2*values_J46 #for crossing at 1
#y = 2*(J35zero-2*values_J46) #for crossing at 2
E2 = np.concatenate((energy_tracker[:64,1], energy_tracker[64:,2]))
E3 = np.concatenate((energy_tracker[:64,2], energy_tracker[64:,1]))
plt.plot(y, energy_tracker[:,0], label ='$E_1$')
#plt.plot(y, energy_tracker[:,1], label ='$E_2$')
#plt.plot(y, energy_tracker[:,2], label ='$E_3$')
plt.plot(y, E2, label ='$E_2$')
plt.plot(y, E3, label ='$E_3$')
plt.plot(y, energy_tracker[:,3], label ='$E_4$')
plt.plot(y, energy_tracker[:,4], label ='$E_5$')
plt.plot(y, energy_tracker[:,5], label ='$E_6$')
plt.plot(y, energy_tracker[:,6], label ='$E_7$')
plt.axvline(x=0, color='k', linestyle='dotted')
#plt.annotate('', xy=(-0.25,-4.7), xytext=(-0.25,-3.65), arrowprops=dict(arrowstyle='<->')) # for step 7A
#plt.annotate('', xy=(-0.5,-5.2), xytext=(-0.5,-4.05), arrowprops=dict(arrowstyle='<->')) # for step 8A
plt.annotate('', xy=(0,-4.2), xytext=(0,-3.5), arrowprops=dict(arrowstyle='<->')) # for step 9A
plt.annotate('', xy=(0,-2.5), xytext=(0,-1.8), arrowprops=dict(arrowstyle='<->')) # for step 9A
plt.legend()
#plt.xlabel('$J_{p1} - J_{p2}$ ($E_0$)') #for step 7A
#plt.xlabel('$J_{35} - J_{46}$ ($E_0$)') #for step 8A
plt.xlabel('$J_{mid} - J_p$ ($E_0$)') #for step 9A
plt.ylabel('energy ($E_0$)')
#plt.title('Energies of the septuplet subspace')
plt.show()

# we plot the differences of succesive eigenstate's energies
plt.figure()
plt.plot(y, E2 - energy_tracker[:,0], label ='$E_2 - E_1$') #this
#plt.plot(y, energy_tracker[:,2] - energy_tracker[:,1], label ='$E_3 - E_2$')
#plt.plot(y, energy_tracker[:,3] - energy_tracker[:,2], label ='$E_4 - E_3$')
plt.plot(y, energy_tracker[:,4] - energy_tracker[:,3], label ='$E_5 - E_4$') #this
#plt.plot(y, energy_tracker[:,5] - energy_tracker[:,4], label ='$E_6 - E_5$')
plt.plot(y, energy_tracker[:,6] - energy_tracker[:,5], label ='$E_7 - E_6$') #this
plt.axvline(x=0, color='k', linestyle='dotted')
plt.legend()
#plt.xlabel('$J_{p1} - J_{p2}$ ($E_0$)') #for step 7A
#plt.xlabel('$J_{35} - J_{46}$ ($E_0$)') #for step 8A
plt.xlabel('$J_{mid} - J_p$ ($E_0$)') #for step 9A
plt.ylabel('energy ($E_0$)')
#plt.title('Energy differences of adjacent states of the septuplet subspace')
plt.show()

# finally we colour plot the expect_tracker matrix data array and the amplitude_tracker for the fourier transform
x = t
plt.pcolormesh(x, y, expect_tracker)
#plt.title('Oscillations in the septuplet subspace')
plt.axhline(y=0, color='w', linestyle='dotted')
plt.xlabel('time ($t_0$)')
#plt.ylabel('$J_{p1} - J_{p2}$ ($E_0$)') #for step 7A
#plt.ylabel('$J_{35} - J_{46}$ ($E_0$)') #for step 8A
plt.ylabel('$J_{mid} - J_p$ ($E_0$)') #for step 9A
plt.colorbar()
plt.show()

# to plot the whole frequency spectrum
x = sample_freq 
plt.pcolormesh(x[:50], y, amplitude_tracker[:,:50])
plt.axhline(y=0, color='w', linestyle='dotted')
#plt.title('Fourier transfom of the oscillations in the septuplet subspace')
plt.xlabel('frequency ($f_0$)')
#plt.ylabel('$J_{p1} - J_{p2}$ ($E_0$)') #for step 7A
#plt.ylabel('$J_{35} - J_{46}$ ($E_0$)') #for step 8A
plt.ylabel('$J_{mid} - J_p$ ($E_0$)') #for step 9A
plt.colorbar()
plt.show()
"""
# ---------------------------------- PROTOCOL B ---------------------------------------

# we plot the energy of the eigenstates
plt.figure()
#y = J35zero-2*values_J46 #for crossing at 1
y = 2*(J35zero-2*values_J46) #for crossing at value
#E3 = np.concatenate((energy_tracker[:85,2], energy_tracker[85:,3])) # for step 8
#E4 = np.concatenate((energy_tracker[:20,4], energy_tracker[20:81,3], energy_tracker[81:,4])) # for step 8
#E5 = np.concatenate((energy_tracker[:20,3], energy_tracker[20:81,4], energy_tracker[81:85,3], energy_tracker[85:,2])) # for step 8
#E1 = np.concatenate((energy_tracker[:33,2], energy_tracker[33:,1])) # for step 9
#E2 = np.concatenate((energy_tracker[:33,3], energy_tracker[33:50,4], energy_tracker[50:,5])) #for step 9
#E3 = np.concatenate((energy_tracker[:50,5], energy_tracker[50:76,4], energy_tracker[76:,3])) # for step 9
plt.plot(y, energy_tracker[:,0], label ='$E_1$')
plt.plot(y, energy_tracker[:,1], label ='$E_2$')
plt.plot(y, energy_tracker[:,2], label ='$E_3$')
plt.plot(y, energy_tracker[:,3], label ='$E_4$')
plt.plot(y, energy_tracker[:,4], label ='$E_5$')
#plt.plot(y, E3, label ='$E_3$')
#plt.plot(y, E4, label ='$E_4$')
#plt.plot(y, E5, label ='$E_5$')
plt.plot(y, energy_tracker[:,5], label ='$E_6$')
plt.plot(y, energy_tracker[:,6], label ='$E_7$')
#plt.plot(y, E1, label ='$E_1$') # step 9
#plt.plot(y, E2, label ='$E_2$') # step 9
#plt.plot(y, E3, label ='$E_3$') # step 9
plt.axvline(x=0, color='k', linestyle='dotted')
plt.annotate('', xy=(0,-5.2), xytext=(0,-4.5), arrowprops=dict(arrowstyle='<->', color='red')) # for step 9B
plt.annotate('', xy=(0,-3.2), xytext=(0,-2.5), arrowprops=dict(arrowstyle='<->', color='red')) # for step 9B
#plt.annotate('', xy=(-0.5,-2.75), xytext=(-0.5,-3.5), arrowprops=dict(arrowstyle='<->', color='red')) # for step 9B
#plt.annotate('', xy=(-0.25,-2.37), xytext=(-0.25,-3.5), arrowprops=dict(arrowstyle='<->', color = 'olive')) # for step 9B
plt.legend()
#plt.xlabel('$J_{c1} - J_{c2}$ ($E_0$)') #for step 5B
#plt.xlabel('$J_{34} - J_{56}$ ($E_0$)') #for step 6B
#plt.xlabel('$J_{12} - J_{78}$ ($E_0$)') #for step 7B
#plt.xlabel('$J_{in} - J_{out}$ ($E_0$)') #for step 8B
plt.xlabel('$J_v - J_h$ ($E_0$)') #for step 9B
plt.ylabel('energy ($E_0$)')
#plt.title('Energies of the septuplet subspace')
plt.show()

# finally we colour plot the expect_tracker matrix data array and the amplitude_tracker for the fourier transform
x = t
plt.pcolormesh(x, y, expect_tracker)
#plt.title('Oscillations in the septuplet subspace')
plt.axhline(y=0, color='w', linestyle='dotted')
plt.xlabel('time ($t_0$)')
#plt.ylabel('$J_{c1} - J_{c2}$ ($E_0$)') #for step 5B
#plt.ylabel('$J_{34} - J_{56}$ ($E_0$)') #for step 6B
#plt.ylabel('$J_{12} - J_{78}$ ($E_0$)') #for step 7B
#plt.ylabel('$J_{in} - J_{out}$ ($E_0$)') #for step 8B
plt.ylabel('$J_v - J_h$ ($E_0$)') #for step 9B
plt.colorbar()
plt.show()

# to plot the whole frequency spectrum
x = sample_freq
plt.pcolormesh(x[:50], y, amplitude_tracker[:,:50])
plt.axhline(y=0, color='w', linestyle='dotted')
#plt.title('Fourier transfom of the oscillations in the septuplet subspace')
#plt.annotate('', xy=(0.5,-0.75), xytext=(1.3,-0.6), arrowprops=dict(arrowstyle='<-', color = 'olive', linewidth = 3)) # for step 9B1
#plt.annotate('', xy=(0.92,-0.2), xytext=(1.3,-0.5), arrowprops=dict(arrowstyle='->', color = 'red', linewidth = 3)) # for step 9B2
plt.xlabel('frequency ($f_0$)')
#plt.ylabel('$J_{c1} - J_{c2}$ ($E_0$)') #for step 5B
#plt.ylabel('$J_{34} - J_{56}$ ($E_0$)') #for step 6B
#plt.ylabel('$J_{12} - J_{78}$ ($E_0$)') #for step 7B
#plt.ylabel('$J_{in} - J_{out}$ ($E_0$)') #for step 8B
plt.ylabel('$J_v - J_h$ ($E_0$)') #for step 9B
plt.colorbar()
plt.show()