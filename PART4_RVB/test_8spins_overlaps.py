#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 10:56:54 2023

@author: ceboncompte
"""

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

egs, gs = H.groundstate()

#We check it belongs to the singlet (yes)
#trans_matrix = coupled_matrix_gen(spins)
#gs.transform(trans_matrix)


# Now we want to compute the several overlaps between this state and all the 10 pairs of singlets
# To do this we will use the partial trace to trace out information of the rest of spins
singlet = (tensor(basis(2,0), basis(2,1)) - tensor(basis(2,1), basis(2,0))).unit()
singlet_dm = singlet*singlet.dag()
#singlet_dm.dims

# We reduce the states
#gs.dims
dm13 = gs.ptrace([0, 2]) #always returns dm
dm35 = gs.ptrace([2, 4])
dm57 = gs.ptrace([4, 6])
dm12 = gs.ptrace([0, 1])
dm34 = gs.ptrace([2, 3])
dm56 = gs.ptrace([4, 5])
dm78 = gs.ptrace([6, 7])
dm24 = gs.ptrace([1, 3])
dm46 = gs.ptrace([3, 5])
dm68 = gs.ptrace([5, 7])
#dm13.dims

# We measure the probability of singlets in each of these positions
p13 = (dm13*singlet_dm).tr()
p35 = (dm35*singlet_dm).tr()
p57 = (dm57*singlet_dm).tr()
p12 = (dm12*singlet_dm).tr()
p34 = (dm34*singlet_dm).tr()
p56 = (dm56*singlet_dm).tr()
p78 = (dm78*singlet_dm).tr()
p24 = (dm24*singlet_dm).tr()
p46 = (dm46*singlet_dm).tr()
p68 = (dm68*singlet_dm).tr()

print(p13, p35, p57, p12, p34, p56, p78, p24, p46, p68)

# Values of J we sweep
number_iterations = 100
Jswept_ini = 0.5
Jswept_fin = 1.5
values_Jswept = np.linspace(Jswept_ini, Jswept_fin, number_iterations)

number_pairs = 10
overlap_tracker = np.zeros((number_iterations, number_pairs))
n = 0
Jzero = 2
B = 0.3 #it seems that not setting the value of B at zero causes a transition in the probability
#measurements for some cases of exchanges

# We do the same sweeping exchanges
for Jswept in values_Jswept:
    print('Computing overlaps... %', n*100/number_iterations)
    """
    Jij_vector = np.array([Jzero - Jswept, 0, 1, 0, 1, 0, Jswept])
    Jij_ladder = np.array([1, 1, 1, 1, 1, 1])
    
    Jij_vector = np.array([Jzero - Jswept, 0, 1.7, 0, 0.3, 0, Jswept])
    Jij_ladder = np.array([0.2, 1.1, 1, 0.5, 1.3, 1.4])
    
    Jij_vector = np.array([Jzero - Jswept, 0, 0.3, 0, 0.5, 0, Jswept])
    Jij_ladder = np.array([0.3, 0.8, 0.3, 0.6, 0.3, 0.4])
    
    Jij_vector = np.array([1, 0, 1, 0, 1, 0, 1])
    Jij_ladder = np.array([1, 1, 2*(Jzero - Jswept), 2*Jswept, 1, 1])
    
    Jij_vector = np.array([Jzero - Jswept, 0, Jzero - Jswept, 0, Jzero - Jswept, 0, Jzero - Jswept])
    Jij_ladder = np.array([Jzero - Jswept, Jzero - Jswept, Jswept, Jswept, Jzero - Jswept, Jzero - Jswept])
    """
    Jij_vector = np.array([1, 0, 1, 0, 1, 0, 1])
    Jij_ladder = np.array([1, 1, Jzero - Jswept, Jswept, 1, 1])
    
    H = hheis_general(Jij_vector, spins, B) + ladder_exchanges(Jij_ladder, spins)
    egs, gs = H.groundstate()
    """
    if n%10 ==0:
        print(egs)
    """
    #for i in range(number_pairs):
    dm13 = gs.ptrace([0, 2]) #always returns dm
    dm35 = gs.ptrace([2, 4])
    dm57 = gs.ptrace([4, 6])
    dm12 = gs.ptrace([0, 1])
    dm34 = gs.ptrace([2, 3])
    dm56 = gs.ptrace([4, 5])
    dm78 = gs.ptrace([6, 7])
    dm24 = gs.ptrace([1, 3])
    dm46 = gs.ptrace([3, 5])
    dm68 = gs.ptrace([5, 7])    
    
    overlap_tracker[n,0] = (dm13*singlet_dm).tr()
    overlap_tracker[n,1] = (dm35*singlet_dm).tr()
    overlap_tracker[n,2] = (dm57*singlet_dm).tr()
    overlap_tracker[n,3] = (dm12*singlet_dm).tr()
    overlap_tracker[n,4] = (dm34*singlet_dm).tr()
    overlap_tracker[n,5] = (dm56*singlet_dm).tr()
    overlap_tracker[n,6] = (dm78*singlet_dm).tr()
    overlap_tracker[n,7] = (dm24*singlet_dm).tr()
    overlap_tracker[n,8] = (dm46*singlet_dm).tr()
    overlap_tracker[n,9] = (dm68*singlet_dm).tr()
    
    n += 1

n = 0

# We plot the results
plt.figure()
for k in range(number_pairs):
    plt.plot(values_Jswept, overlap_tracker[:,k], label =f'$S{k+1}$')
plt.axvline(x=1, color='k', linestyle='dotted')
plt.legend()
plt.xlabel('$J$ ($E_0$)')
plt.ylabel('probability of singlet')
plt.title('Singlet overlaps')
plt.show()