#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 09:47:54 2022

@author: ceboncompte
"""

from qutip import *
import numpy as np

def A(j1, j2, j): # Called in CG
    
    """First term of the Clebsh Gordan coefficients."""
    
    T1 = np.math.factorial(j1 + j2 - j)
    T2 = np.math.factorial(j + j1 - j2)
    T3 = np.math.factorial(j2 + j - j1)
    T4 = np.math.factorial(j1 + j2 + j + 1)
    return (T1*T2*T3)/T4

def B(j1, j2, j, m1, m2, m): # Called in CG
    
    """Second term of the Clebsh Gordan coefficients."""
    
    if j1 + m1 >= 0 and j1 - m1 >= 0 and j2 + m2 >= 0 and j2 - m2 >= 0 and j + m >= 0 and j - m >= 0:
        T1 = np.math.factorial(j1 + m1)
        T2 = np.math.factorial(j1 - m1)
        T3 = np.math.factorial(j2 + m2)
        T4 = np.math.factorial(j2 - m2)
        T5 = np.math.factorial(j + m)
        T6 = np.math.factorial(j - m)
        return T1*T2*T3*T4*T5*T6
    else:
        return 0

def C(j1, j2, j, m1, m2, v): # Called in CG
    
    """Third term term of the Clebsh Gordan coefficients."""
    
    T1 = np.math.factorial(j1 + j2 - j - v)
    T2 = np.math.factorial(j1 - m1 - v)
    T3 = np.math.factorial(j2 + m2 - v)
    T4 = np.math.factorial(j - j2 + m1 + v)
    T5 = np.math.factorial(j - j1 - m2 + v)
    return T1*T2*T3*T4*T5

def CG(j1, j2, j, m1, m2, m): # Called in coupled_basis_matrix
    
    """Gives Clebsh Gordan coefficient for two given spins in the computational basis (total spin and z-axis projection needed for both spins)."""
    
    if m == m1+m2:
        T1 = ((2*j + 1)*A(j1, j2, j)*B(j1, j2, j, m1, m2, m))**0.5
        T2 = 0
        for v in range(8): # This needs to be revised
            if j1 + j2 - j - v >= 0 and j1 - m1 - v >= 0 and j2 + m2 - v >= 0 and j - j2 + m1 + v >= 0 and j - j1 - m2 + v >= 0:
                T2 += (((-1)**v)*(C(j1, j2, j, m1, m2, v))**(-1))/np.math.factorial(v)
            else:
                continue
        return T1*T2
    else:
        return 0

def compose(j1, j2): # Called in label_rec
    
    """Returns a list with ordered (from lower to higher) possible total spins in the 2-spin composition."""
    
    list12 = np.array([])

    n12 = 0
    while j1+j2-n12 >= 0 and j1+j2-n12 >= j1-j2:
        element12 = j1+j2-n12
        list12 = np.append(list12, element12)
        n12 += 1
    list12 = np.sort(list12)
    return list12

def label_rec(spins): # Called in js_ms_labels
    
    """Gives a list of the total-J labels required for an arbitrary number of 1/2 spin compositions in a chain."""
    
    j = 1/2
    
    if spins == 1:
        labelsj = np.ones((1, ))*0.5
    else:
        labels_pre = label_rec(spins-1)
        labelsj = np.array([])
        
        for element in labels_pre:
            compose_single = compose(element, j)
            labelsj = np.concatenate((labelsj, compose_single), axis = None)
    
    return labelsj

def js_ms_labels(spins): # Called in labels_sorted
    
    """Gives a list of all the labels required for an arbitrary number of 1/2 spin compositions in a chain."""
    
    labelst = np.zeros((spins, 2**spins))
    
    totalspins = label_rec(spins)
    projection_vector = np.array([])
    
    for element_totalspins in totalspins:
        count = -element_totalspins
        while count <= element_totalspins:
            projection_vector = np.append(projection_vector, count)
            count += 1
    
    labelst[-1,:] = projection_vector
    repeat_list = totalspins*2 + 1
    repeat_list = repeat_list.astype(int)
    reb = np.repeat(totalspins, repeat_list)
    labelst[-2,:] = reb
    
    for i in range(spins-2):
        comp_from_lower = label_rec(i+2) # this has to be always label_rec(2)
        repeat_list = comp_from_lower*2 + 1
        repeat_list = repeat_list.astype(int)
        reb = np.repeat(comp_from_lower, repeat_list)
        arr2 = np.ones(len(reb), )*(2**(spins-2-i))
        arr2 = arr2.astype(int)
        composition = np.repeat(reb,arr2)
        labelst[i,:] = composition
    
    return np.transpose(labelst)

def labels_sorted(spins): # Called in coupled_matrix_gen
    
    """Sorts the labels list so we can use it to basis-transform to a clear block diagonal matrix."""
    
    jm_unsorted = js_ms_labels(spins)
    # we first sort by total J angular momentum
    sorted_half = jm_unsorted[np.argsort(jm_unsorted[:, -2])] # -2 respresents the before-last column
    # now we sort the same-total-J blocks by projection
    cut = 0
    sorted_full = np.zeros((2**spins, spins))
    for k in range(2**spins):
        if k != (2**spins-1):
            if sorted_half[k, -2] != sorted_half[k+1, -2]:
                block = sorted_half[cut:k+1,:]
                block = block[np.argsort(block[:, -1])]
                sorted_full[cut:k+1,:] = block
                cut = k+1
        if k == (2**spins-1):
            block = sorted_half[cut:,:]
            block = block[np.argsort(block[:, -1])]
            sorted_full[cut:,:] = block
            cut = k+1
    return sorted_full

def coupled_matrix_gen(spins):
    
    """Gives the coupled basis matrix to transform to it from the computational basis."""
    
    labels = labels_sorted(spins)
    #print(labels)
    # we add an extra column in front full of 0.5s
    labels = np.hstack((np.ones((2**spins,1))*(1/2), labels))
    j_second = 1/2
    fourspinmat = np.zeros((2**spins, 2**spins), dtype = complex)
    
    for i in range(2**spins): # Vector index
        eigenvector = 0 # We reset the eigenvector
        
        for j in range(2**spins): # All computational m combinations loop
            bit_decimal = np.array([[j]], dtype=np.uint8)
            labels_comput = np.flipud((np.unpackbits(bit_decimal, axis = 1, count=spins, bitorder = 'little')[0] -1/2)*(-1))
            
            eigenvector_coef = 1
            m_first = 0
            for k in range(spins-1):
                if k != spins-2:
                    m_first += labels_comput[k]
                    m_second = labels_comput[k+1]
                    m_composed = m_first + m_second
                    j_first = labels[i,k]
                    j_composed = labels[i,k+1]
                    eigenvector_coef *= CG(j_first, j_second, j_composed, m_first, m_second, m_composed)
                else:
                    m_first += labels_comput[k]
                    m_second = labels_comput[k+1]
                    m_composed = labels[i,k+2]
                    j_first = labels[i,k]
                    j_composed = labels[i,k+1]
                    eigenvector_coef *= CG(j_first, j_second, j_composed, m_first, m_second, m_composed)
                
            eigenvector += eigenvector_coef*np.array(basis(2**spins, j).trans())[0]
        
        fourspinmat[i,:] = eigenvector
    return fourspinmat