#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 20:07:47 2023

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

# In this case we attempt to check if a tuning-square protocols works with a full ladder of exchanges
# It apears that if the mid couplings are low enough, like 2 orders of magnitude low it works

# --------------------------------------------- FOR THE ENERGY PLOTS
number_iterations = 400 # number of different values of J we test
number_mid_values = 50
energy_tracker = np.zeros((number_iterations, 7)) # septuplet refers to the number of states in the subspace
gap_tracker = np.zeros((number_iterations, number_mid_values))
max_tracker = np.zeros((number_mid_values, 2))
J46ini = 1.5
J46fin = 0.5
values_J46 = np.linspace(J46ini, J46fin, number_iterations)
acc = values_J46[0]-values_J46[1]
print(acc) # put an upper bound of this to the errorbars
#values_Jmid = np.linspace(0, 0.3, number_mid_values)
values_Jmid = np.logspace(-2, 0, number_mid_values)
n = 0
k = 0
J35zero = 2

# We generate the basis-transformation matrix
trans_matrix = coupled_matrix_gen(spins)
for Jmid in values_Jmid:
    print('constructing plots...', (k/number_mid_values)*100, "%")
    for J46 in values_J46:
        # We first want to create the arbitrary Hamiltonian
        Jij_vector = np.array([0.6, 0, 0.6, 0, 0.8, 0, 0.7])
        Jij_ladder = np.array([J35zero - J46, J46, Jmid, Jmid, 1, 1.1])
        
        #Jij_vector = np.array([J35zero - J46, 0, J46, 0, 1, 0, 1])
        #Jij_ladder = np.array([1.2, 0.7, 0.01, 0.01, 1, 1])
        spins = len(Jij_vector) + 1
        H = hheis_general(Jij_vector, spins, B) + ladder_exchanges(Jij_ladder, spins)
        
        # We basis transform the Hamiltonian matrix
        H_coup = basis_transformation(H, trans_matrix)
        
        # We take the first septuplet and we get the energies of the states (end)
        H_sept = H_coup[198:205, 198:205]
        
        # We diagonalize and obtain energy values
        ev = sp.linalg.eigvalsh(H_sept)
        energy_tracker[n,:] = ev
        
        n+=1
    
    n = 0
    #max_tracker[k,:] = np.max(energy_tracker[:,0]), J35zero-2*values_J46[argm]
    gap_tracker[:,k] = energy_tracker[:,5] - energy_tracker[:,1]
    argm = np.argmin(gap_tracker[:,k])
    max_tracker[k,:] = np.min(gap_tracker[:,k]), J35zero-2*values_J46[argm]
    
    #plt.plot(2*(1-values_J46), gap_tracker[:,k])
    #plt.show()
    
    k+=1

k = 0

# we plot the position of the feature
plt.figure()
plt.errorbar(values_Jmid, max_tracker[:,1], yerr = 0.003, fmt ='b_')
plt.axhline(y=0, color='k', linestyle='dotted')
#plt.legend()
plt.xscale('log')
plt.xlabel('$J_{mid}$ ($E_0$)')
plt.ylabel('$J_{p1} - J_{p2}$ ($E_0$)')
#plt.title('Deviation of the crossing feature')
plt.show()

for i in range(number_mid_values):
    if i%12 == 0:
        plt.plot(2*(1-values_J46), gap_tracker[:,i], label ='$J_{mid} = $' + f'{values_Jmid[i].round(2)}' )
plt.axvline(x=0, color='k', linestyle='dotted')
plt.legend()
plt.xlabel('$J_{p1} - J_{p2}$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
#plt.title('Energy gap $E2 - E1$')
plt.show()