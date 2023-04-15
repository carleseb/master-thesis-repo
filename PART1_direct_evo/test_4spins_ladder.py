#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 14:50:51 2022

@author: ceboncompte
"""

from DM_solver.solver  import H_channel, H_solver

from hamiltonian import hheis_general, heisenberg_hamiltonian_4, ladder_exchanges, is_unitary
from use import matrix_plot, basis_transformation
from basis_matrix import coupled_matrix_gen

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# We first want to create the arbitrary Hamiltonian and print the matrix
Jij_vector = np.array([1, 0, 1])
Jij_ladder = np.array([1, 1])
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

# Future: look at energies target state: it seems that here middle triplet
# shows equal spacing for all states for B=0.3
"""
# We get the eigenvalues
e = sp.linalg.eigvalsh(H_coup)
print(e)
"""

# We want to see the energies of the eigenstates for this case, so we set a low magnetic field (also works with 0) and
# compute-plot them for different values of J
# We also want to see the oscillationss of the input states for this subspace
# We compute the initial state (Tminus, S) in the computational basis and we basis-transform it
S = (basis(4,1) - basis(4,2)).unit()
T0 = (basis(4,1) + basis(4,2)).unit()
Tplus = basis(4,0)
Tminus = basis(4,3)

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
minus_1 = TM1_t.eliminate_states([0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
print(np.array(minus_1))

SS = tensor(S,S)
SS_t = basis_transformation(SS, trans_matrix)
print(SS_t)

S13T24 = (basis(16,7) - basis(16,13)).unit()
S13T24_t = basis_transformation(S13T24, trans_matrix)
print(S13T24_t)
horizontal_0 = S13T24_t.eliminate_states([0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
print(np.array(horizontal_0))

T13S24 = (basis(16,11) - basis(16,14)).unit()
T13S24_t = basis_transformation(T13S24, trans_matrix)
horizontal_1 = T13S24_t.eliminate_states([0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
print(np.array(horizontal_1))

# Intitial state is the second state of the first triplet of the basis-transformation matrix, so we get its density matrix
#ket0 = basis(3, 1)
ket0 = minus_0
dm0 = ket0 * ket0.dag()
dm0 = np.array(dm0)
    
# If initial state/measured state is minus_1 change
ket1 = minus_1
dm1 = ket1 * ket1.dag()
dm1 = np.array(dm1)

# Other states
keth0 = horizontal_0
dmh0 = keth0 * keth0.dag()
dmh0 = np.array(dmh0)

keth1 = horizontal_1
dmh1 = keth1 * keth1.dag()
dmh1 = np.array(dmh1)

# --------------------------------------------- FOR THE ENERGY PLOTS
number_iterations = 100 # number of different values of J we test
energy_tracker = np.zeros((number_iterations, 3)) # 3 (triplet) refers to the number of states in the subspace
J34ini = 1.5
J34fin = 0.5
values_J34 = np.linspace(J34ini, J34fin, number_iterations)
#values_J34 = np.linspace(2, 0, number_iterations)
n = 0

# --------------------------------------------- FOR THE OSCILLATIONS AND FOURIERS
end_time = 15 # time of the oscillations (in natural units)
dimensions_expect_tracker = 1000 # number of time steps
expect_tracker = np.zeros((number_iterations, dimensions_expect_tracker))
amplitude_tracker = np.zeros((number_iterations, dimensions_expect_tracker))
J12zero = 2

# We generate the basis-transformation matrix
trans_matrix = coupled_matrix_gen(spins)

for J34 in values_J34:
    # We first want to create the arbitrary Hamiltonian and print the matrix  
    #Jij_ladder = np.array([0.5, 0.8]) #First step
    #Jij_ladder = np.array([J12zero - J34, J34])
    
    Jij_vector = np.array([1.5*(J12zero-J34), 0, 1.5*J34]) #Second step
    Jij_ladder = np.array([1, 1])
    spins = len(Jij_vector) + 1
    
    H = hheis_general(Jij_vector, spins, B) + ladder_exchanges(Jij_ladder, spins)
    
    # We plot the matrix
#    matrix_plot(H)
    
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
    
    # calculate for end_time
    number_steps = dimensions_expect_tracker
    sample_rate = number_steps/end_time
    calculation.calculate(dmh0, end_time = end_time, sample_rate = sample_rate)
    
    # calculate the expectatoin value of the matrix dm0 and plot
    # plus fourier transform
    dm0_expect = calculation.return_expectation_values(dmh1)
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
    ev = sp.linalg.eigvalsh(H_triplet)
    energy_tracker[n,:] = ev
    n+=1

# For the first step
# we plot the energy of the eigenstates
plt.figure()
plt.plot(J12zero-2*values_J34, np.transpose(energy_tracker)[0,:], label ='$E_1$')
plt.plot(J12zero-2*values_J34, np.transpose(energy_tracker)[1,:], label ='$E_2$')
plt.plot(J12zero-2*values_J34, np.transpose(energy_tracker)[2,:], label ='$E_3$')
plt.axvline(x=0, color='k', linestyle = 'dotted')
plt.legend()
plt.xlabel('$J_{13} - J_{24}$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
#plt.title('Energies of the triplet subspace')
plt.show()

# we plot the differences of exchanges
plt.plot(J12zero-2*values_J34, np.transpose(energy_tracker)[1,:] - np.transpose(energy_tracker)[0,:], label ='$E_2 - E_1$')
plt.plot(J12zero-2*values_J34, np.transpose(energy_tracker)[2,:] - np.transpose(energy_tracker)[1,:], label ='$E_3 - E_2$')
plt.axvline(x=0, color='k', linestyle = 'dotted')
plt.legend()
plt.xlabel('$J_{13} - J_{24}$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
#plt.title('Energy differences of the adjacent eigenvectors')
plt.show()

# finally we colour plot the expect_tracker matrix data array and the amplitude_tracker for the fourier transform
x = t
y = -(J12zero-2*values_J34) #minus sign for the appropiate convesion of mesh display
plt.pcolormesh(x, y, expect_tracker)
#plt.title('Oscillations in the triplet subspace')
plt.axhline(y=0, color='k', linestyle = 'dotted')
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{13} - J_{24}$ ($E_0$)')
plt.colorbar()
plt.show()

# to plot the whole frequency spectrum (with negative ones)
x = sample_freq
y = -(J12zero - 2*values_J34)
plt.pcolormesh(x[:80], y, amplitude_tracker[:,:80])
#plt.title('Fourier transform of the oscillations in the triplet subspace')
plt.axhline(y=0, color='w', linestyle = 'dotted')
plt.xlabel('frequency ($f_0$)')
plt.ylabel('$J_{13} - J_{24}$ ($E_0$)')
plt.colorbar()
plt.show()

"""
# Uncomment this if second step
# we plot the energy of the eigenstates
y = 1.5*(J12zero-2*values_J34)
plt.figure()
plt.plot(y, np.transpose(energy_tracker)[0,:], label ='$E_1$')
plt.plot(y, np.transpose(energy_tracker)[1,:], label ='$E_2$')
plt.plot(y, np.transpose(energy_tracker)[2,:], label ='$E_3$')
plt.axvline(x=0, color='k', linestyle = 'dotted')
plt.legend()
plt.xlabel('$J_{12} - J_{34}$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
#plt.title('Energies of the triplet subspace')
plt.show()

# finally we colour plot the expect_tracker matrix data array and the amplitude_tracker for the fourier transform
x = t
plt.pcolormesh(x, y, expect_tracker)
#plt.title('Oscillations in the triplet subspace')
plt.axhline(y=0, color='k', linestyle = 'dotted')
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{12} - J_{34}$ ($E_0$)')
plt.colorbar()
plt.show()

# to plot the whole frequency spectrum (with negative ones)
x = sample_freq
plt.pcolormesh(x[:80], y, amplitude_tracker[:,:80])
#plt.title('Fourier transform of the oscillations in the triplet subspace')
plt.axhline(y=0, color='w', linestyle = 'dotted')
plt.xlabel('frequency ($f_0$)')
plt.ylabel('$J_{12} - J_{34}$ ($E_0$)')
plt.colorbar()
plt.show()
"""






# Now we take the value for which we observe minimum energy gap between bot triplets (and homogeneous exchange)
# and we fix the J12 and J34 values, varying the J13 and J24 values
# --------------------------------------------- FOR THE ENERGY PLOTS
number_iterations = 100 # number of different values of J we test
energy_tracker_ = np.zeros((number_iterations, 3)) # 3 (triplet) refers to the number of states in the subspace
J24ini = 1.5
J24fin = 0.5
values_J24 = np.linspace(J24ini, J24fin, number_iterations)
#values_J24 = np.linspace(2, 0, number_iterations)
n = 0

# --------------------------------------------- FOR THE OSCILLATIONS AND FOURIERS
end_time = 30 # time of the oscillations (in natural units)
dimensions_expect_tracker = 1000 # number of time steps
expect_tracker = np.zeros((number_iterations, dimensions_expect_tracker))
amplitude_tracker = np.zeros((number_iterations, dimensions_expect_tracker))
J13zero = 2
"""
J24ini = 0.5
J24fin = 1.5
"""

# We generate the basis-transformation matrix
trans_matrix = coupled_matrix_gen(spins)

for J24 in values_J24:
    # We first want to create the arbitrary Hamiltonian and print the matrix
    #Jij_vector = np.array([2-1, 0, 1])    
    #Jij_ladder = np.array([J24, J24])
    
    Jij_vector = np.array([J24, 0, J24])
    Jij_ladder = np.array([J13zero - J24, J13zero - J24])
    spins = len(Jij_vector) + 1
    """
    a = J24-1
    Jx = 1+a
    Jy = 1+(3/2*a)
    Jij_vector = np.array([Jx, 0, Jx])    
    Jij_ladder = np.array([Jy, Jy])
    spins = len(Jij_vector) + 1
    """
    
    H = hheis_general(Jij_vector, spins, B) + ladder_exchanges(Jij_ladder, spins)
    
    # We plot the matrix
#    matrix_plot(H)
    
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
    
    # calculate for end_time
    number_steps = dimensions_expect_tracker
    sample_rate = number_steps/end_time
    calculation.calculate(dm0, end_time = end_time, sample_rate = sample_rate)
    # This dm0 is the initial state!
    
    # calculate the expectatoin value of the matrix dm0 and plot
    # plus fourier transform
    dm0_expect = calculation.return_expectation_values(dm1)
    # This dm0/dm1/dmh is the state we measure!
    expect_tracker[n,:] = dm0_expect[0]
    t = calculation.return_time()
    
    # fourier ttransform
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
plt.plot(J13zero-2*values_J24, energy_tracker_[:,0], label ='$E_1$')
plt.plot(J13zero-2*values_J24, energy_tracker_[:,1], label ='$E_2$')
plt.plot(J13zero-2*values_J24, energy_tracker_[:,2], label ='$E_3$')
plt.legend()
plt.axvline(x=0, color='k', linestyle='dotted')
plt.xlabel('$J_{13} - J_{24}$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
#plt.title('Energies of the triplet subspace')
plt.show()

# we arrange the enigenvectors properly (proper plot)
length = len(energy_tracker_[:,1])
a = int(length/2)
E1 = np.concatenate((energy_tracker_[:a,1], energy_tracker_[a:,2]))
E2 = np.concatenate((energy_tracker_[:a,2], energy_tracker_[a:,1]))
#we plot
plt.plot(values_J24, np.transpose(energy_tracker_)[0,:], label ='$E_1$')
plt.plot(values_J24, E1, label ='$E_2$')
plt.plot(values_J24, E2, label ='$E_3$')
plt.legend()
plt.axvline(x=1, color='k', linestyle='dotted')
plt.xlabel('$J_{d1}$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
#plt.title('Energies of the triplet subspace')
plt.show()

# Uncomment if antisymmetric sweep
"""
# we arrange the enigenvectors properly (proper plot)
length = len(energy_tracker_[:,1])
a = int(length/2)
E1 = np.concatenate((energy_tracker_[:a,1], energy_tracker_[a:,2]))
E2 = np.concatenate((energy_tracker_[:a,2], energy_tracker_[a:,1]))
#we plot
plt.plot(J13zero-2*values_J24, np.transpose(energy_tracker_)[0,:], label ='$E_1$')
plt.plot(J13zero-2*values_J24, E1, label ='$E_2$')
plt.plot(J13zero-2*values_J24, E2, label ='$E_3$')
plt.legend()
plt.axvline(x=0, color='k', linestyle='dotted')
plt.xlabel('$J_{d1} - J_{d2}$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
#plt.title('Energies of the triplet subspace')
plt.show()
"""

# finally we colour plot the expect_tracker matrix data array and the amplitude_tracker for the fourier transform
# plot for normal sweep
x = t
#y = np.linspace(J13zero-2*J24ini, J13zero-2*J24fin, number_iterations)
y = np.linspace(J24ini, J24fin, number_iterations)
plt.pcolormesh(x, y, expect_tracker)
#plt.title('Oscillations in the triplet subspace')
plt.axhline(y=1, color='k', linestyle = 'dotted')
plt.xlabel('time ($t_0$)')
plt.ylabel('$J_{13} - J_{24}$ ($E_0$)')
plt.colorbar()
plt.show()

# to plot the whole frequency spectrum (with negative ones)
x = sample_freq
y = np.linspace(J24ini, J24fin, number_iterations)
plt.pcolormesh(x[:80], y, amplitude_tracker[:,:80])
#plt.title('Fourier transform of the oscillations in the triplet subspace')
plt.axhline(y=1, color='w', linestyle = 'dotted')
plt.xlabel('frequency ($f_0$)')
plt.ylabel('$J_{13} - J_{24}$ ($E_0$)')
plt.colorbar()
plt.show()