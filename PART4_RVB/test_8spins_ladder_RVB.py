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

gs = H_coup.groundstate()
print(gs[0], gs[1])

# Now we want to see how the energy of the ground state evolves with the magnetic field
number_iterations = 100
Bini = 0
Bfin = 2
values_B = np.linspace(Bini, Bfin, number_iterations)
energy_tracker = np.zeros((number_iterations))
n = 0

for B in values_B:
    # We first want to create the arbitrary Hamiltonian and print the matrix
    Jij_vector = np.array([1, 0, 1, 0, 1, 0, 1])
    Jij_ladder = np.array([1, 1, 1, 1, 1, 1])
    spins = len(Jij_vector) + 1
    H = hheis_general(Jij_vector, spins, B) + ladder_exchanges(Jij_ladder, spins)
    
    # We get the ground state
    gs = H.groundstate()
    #print(gs[0], gs[1])    
    energy_tracker[n] = gs[0]
    n += 1

# we plot the energy of the groundstate
#plt.figure(figsize=(6,5))
plt.figure()
plt.plot(values_B, energy_tracker, label='E1')
plt.xlabel('$B$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
plt.title('Energy of the groundstate')
plt.show()

# We will try to see how the energy of the ground state behaves around J homogeneous
Jini = 0.5
Jfin = 1.5
values_J = np.linspace(Jini, Jfin, number_iterations)
n = 0

for J in values_J:
    # We first want to create the arbitrary Hamiltonian and print the matrix
    """
    Jij_vector = np.array([1, 0, 1, 0, 1, 0, 1])
    Jij_ladder = np.array([1, 1, 2-J, J, 1, 1])
    """
    
    Jij_vector = np.array([1.4, 0, 0.6, 0, 1, 0, 1.2])
    Jij_ladder = np.array([0.7, 0.8, 2-J, J, 1.3, 1])
    spins = len(Jij_vector) + 1
    H = hheis_general(Jij_vector, spins, B) + ladder_exchanges(Jij_ladder, spins)
    
    # We get the ground state
    gs = H.groundstate()
    #print(gs[0], gs[1])    
    energy_tracker[n] = gs[0]
    n += 1
    
# we plot the energy of the groundstate
#plt.figure(figsize=(6,5))
plt.figure()
plt.plot(values_J, energy_tracker, label='E1')
plt.axvline(x = 1, color = 'k')
plt.xlabel('$J (any)$ ($E_0$)')
plt.ylabel('energy ($E_0$)')
plt.title('Energy of the groundstate')
plt.show()

# Therefore setting B = 0.5 allows us to stay in the not fully polarized state
B = 0.5
end_time = 20 # time of the oscillations (in natural units)
dimensions_time_evo_tracker = 1000 # number of time steps
time_evo_tracker = np.zeros((5, dimensions_time_evo_tracker))

# Now we want to see how does an initial state made out of one of the valence bonds of the
# ground state evolve. In particular we are interested in the probability of measuring all
# the valence bonds composing that ground state
Jij_vector = np.array([1, 0, 1, 0, 1, 0, 1])
Jij_ladder = np.array([1, 1, 1, 1, 1, 1])
spins = len(Jij_vector) + 1
H = hheis_general(Jij_vector, spins, B) + ladder_exchanges(Jij_ladder, spins)
gstate = H.groundstate()
print(gstate[1])

gstate_t = basis_transformation(gstate[1], trans_matrix)
print(gstate_t)

for index, element in enumerate(gstate[1]):
    if element != 0.:
        print(index, element)

# Let's try to reconstruct this ground state
p1 = (basis(16, 5) - basis(16,6) - basis(16,9) + basis(16,10)).unit() #S12, S34
print(p1)
S = (basis(4,1) - basis(4,2)).unit()
#p11 = tensor(S, S) #S12, S34
#print(p11)
p2 = (basis(16, 3) - basis(16,5) - basis(16,10) + basis(16,12)).unit() # S14 S23
print(p2) #S14 S23

p3 = (basis(16, 3) - basis(16,6) - basis(16,9) + basis(16,12)).unit() #S13 S24
print(p3) # not used

c1 = tensor(p1, p1)
c2 = tensor(p3, p1)
c3 = tensor(p1, p3)
c4 = tensor(S, p3, S)
c5 = tensor(p3, p3)
c4.dims = [[16, 16], [1, 1]]

# Let's build a full superposition
state = (c1 + c2 + c3 + c4 + c5).unit()
print(state)

for index, element in enumerate(state):
    if element != 0.:
        print(index, element)

# Anyways, we keep
# We base transform, to make the time evolution cheaper
trans_matrix = coupled_matrix_gen(spins)
H_coup = basis_transformation(H, trans_matrix)
H_singlet = H_coup[0:14, 0:14]

CH0 = H_channel(H_singlet)
CH0.pulse.add_constant(2*np.pi*1.)
# We solve the Schrodinger equation
calculation = H_solver()
calculation.add_channels(CH0)

# Intitial state is the first state (first row) of the basis-transformation matrix
ket0 = tensor(S, S, S, S)
print(ket0)
ket0_t = basis_transformation(ket0, trans_matrix)
print(ket0_t) #this is basis(2,0)
ket0_red = basis(14,0)
dm0 = ket0_red * ket0_red.dag()
dm0 = np.array(dm0)
c1dm = dm0

# The other states composing the RVB (not all of them) are c1, c2, c3, c4 and c5
# However, we have to base transform them and reduce
c2_t = basis_transformation(c2, trans_matrix)
print(c2_t)
c2_red = (basis(14,0) + np.sqrt(3)*basis(14,1)).unit()
print(c2_red)
c2dm = c2_red * c2_red.dag()
c2dm = np.array(c2dm)

c3_t = basis_transformation(c3, trans_matrix)
print(c3_t)
c3_red = (basis(14,0) + np.sqrt(3)*basis(14,13)).unit()
print(c3_red)
c3dm = c3_red * c3_red.dag()
c3dm = np.array(c3dm)

c4_t = basis_transformation(c4, trans_matrix)
print(c4_t)
c4_red = (basis(14,0) + np.sqrt(3)*basis(14,10)).unit()
print(c4_red)
c4dm = c4_red * c4_red.dag()
c4dm = np.array(c4dm)

c5_t = basis_transformation(c5, trans_matrix)
print(c5_t)
c5_red = (basis(14,0) + np.sqrt(3)*basis(14,1) + 3*basis(14,12) + np.sqrt(3)*basis(14,13)).unit()
print(c5_red)
c5dm = c5_red * c5_red.dag()
c5dm = np.array(c5dm)

# Calculate for end_time
number_steps = dimensions_time_evo_tracker
sample_rate = number_steps/end_time
calculation.calculate(dm0, end_time = end_time, sample_rate = sample_rate)

# Calculate the expectatoin value of the matrix dm0 and plot
dm_expect = calculation.return_expectation_values(c1dm, c2dm, c3dm, c4dm, c5dm)
time_evo_tracker[0,:] = dm_expect[0]
time_evo_tracker[1,:] = dm_expect[1]
time_evo_tracker[2,:] = dm_expect[2]
time_evo_tracker[3,:] = dm_expect[3]
time_evo_tracker[4,:] = dm_expect[4]
t = calculation.return_time()

# Finally we plot the oscillations
x = t
#plt.plot(x, time_evo_tracker[0,:], label='$\Psi_0$')
#plt.plot(x, time_evo_tracker[1,:], label='$\Psi_1$')
#plt.plot(x, time_evo_tracker[2,:], label='$\Psi_2$')
#plt.plot(x, time_evo_tracker[3,:], label='$\Psi_3$')
plt.plot(x, time_evo_tracker[4,:], label='$\Psi_4$')
plt.title('Probability of measurement')
plt.xlabel('time ($t_0$)')
plt.ylabel('probability')
plt.legend()
#plt.ylabel('$J_{12} - J_{23}$ ($E_0$)')
plt.show()