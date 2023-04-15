#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 16:40:56 2023

@author: ceboncompte
"""

from DM_solver.solver  import H_channel, H_solver
from DM_solver.pulse_generation.pulse import pulse

from DM_solver.utility.pauli import X,Y,Z

import matplotlib.pyplot as plt
import numpy as np

from qutip import *

'''
Example 1 : set zeeman difference and drive qubit
'''
f_qubit = 1e9
f_drive = 10e6

# define channel for that sets the energy separation -- Sz hamiltonian. Units of [rad]
Qubit1_Z = H_channel(Z/2)
Qubit1_Z.pulse.add_constant(2*np.pi*f_qubit)

# define channel that drives the qubit -- Sx hamiltonian.
Qubit1_X = H_channel(X/2)
# define a pulse from 10 ns to 60ns
#Qubit1_X.pulse.add_MW_pulse(20e-9,70e-9, amp=2*np.pi*f_drive, freq=f_qubit, phase=np.pi/2) #original
Qubit1_X.pulse.add_MW_pulse(20e-9,70e-9, amp=2*np.pi*f_drive, freq=f_qubit, phase=np.pi/2)

# show the pulse
#Qubit1_X.plot_pulse(t_end=100e-9, sample_rate=1e11) #original
Qubit1_X.plot_pulse(t_end=150e-9, sample_rate=1e11)

# make object that solves the schrodinger equation
calculation = H_solver()
calculation.add_channels(Qubit1_Z, Qubit1_X)

# initial density matrix
psi_0 = np.matrix([[1,0],[0,0]])

# calculate for 100ns with time steps of 10ps
calculation.calculate(psi_0, end_time = 150e-9, sample_rate = 1e11)

# calculate some expectatoin values and plot
Z_expect, X_expect = calculation.return_expectation_values(Z, X)
t = calculation.return_time()
plt.plot(t, Z_expect)
plt.show()
plt.plot(t, X_expect)
plt.show()

# now we want to obtain the final density matrix
u = calculation.get_unitary()
u = Qobj(u)
rho = u*psi_0*u.dag()
one = basis(2,1)

print(rho, rho*rho, np.allclose(rho, rho*rho)) #shows final rho is pure
print(np.trace(rho), np.trace(rho*rho)) # also shows that
print(np.trace(rho*(one*one.dag()))) # shows expectation value of ket one

'''
We may do the same with QuTiP
'''

# For quantum states we use sesolve (only expectation values)
H = 2*np.pi * 0.1 * sigmax()
psi0 = basis(2, 0)
times = np.linspace(0.0, 10.0, 20)
#4th argument is a list of operator for which we want the expectation values
result = sesolve(H, psi0, times, [sigmaz()])
result.solver
result.expect
result.states
plt.plot(times, result.expect[0])
plt.show()

# For density matrices we use mesolve
H = 2*np.pi * 0.1 * sigmax()
psi0 = basis(2, 0)
times = np.linspace(0.0, 10.0, 20)
#Now 4th argument is a list of COLLAPSE operators
#5th argument is a list of operator for which we want the expectation values (if empty returns the state vectors)
result = mesolve(H, psi0, times, [0.5*sigmax()], [sigmaz()])
result.solver
result.expect
result.states
plt.plot(times, result.expect[0])
plt.show()

# To get the states
H = 2*np.pi * 0.1 * sigmax()
psi0 = basis(2, 0)
times = np.linspace(0.0, 10.0, 20)
#Now 4th argument is a list of COLLAPSE operators
#5th argument is a list of operator for which we want the expectation values (if empty returns the state vectors)
result = mesolve(H, psi0, times, [0.5*sigmax()], [])
result.solver
result.expect
result.states

#Let's inspect result.states (we use mesolve with no collapse operators)
H = 2*np.pi * 0.1 * sigmax()
psi0 = basis(2, 0)
times = np.linspace(0.0, 10.0, 20)
#Now 4th argument is a list of COLLAPSE operators
#5th argument is a list of operator for which we want the expectation values (if empty returns the state vectors)
result = mesolve(H, psi0, times, [], [])
result.solver
result.expect
result.states
#inspect
result.states[0]
result.states[0].norm()
result.states[0].data
result.states[0].full #?
print(np.array(result.states[0])) #this stores the vector matrix-correctly (column)
result.states[0]*result.states[0].dag() #this can be used to obtain density matrix
len(result.states)

states = np.zeros((1, len(result.states), 2, 1))
for k in range(len(result.states)):
    states[0,k,:,:]=np.array(result.states[k])

print(states) # this is how we store states
