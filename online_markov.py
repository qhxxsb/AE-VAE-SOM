#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
	Author : Sean Zhong, July 2021

"""
#%%
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from termcolor import colored
from colorama import Fore, Style

"""
	An Empirical Markov Chain State Transition Matrix object
	All states are labeled as 0,1,2,... 
"""

class MCTransitionMatrix(object):
    def __init__(self, num_of_possible_states: np.int64, U=None, AC=None, states=None):
        self._N0 = num_of_possible_states
        assert self._N0 > 1

        self._P0 = np.eye(self._N0)
        self._IC = np.ones(self._N0, dtype=np.int64)
        
        if U is not None:
            self._U = U
            assert self._U.ndim == 2
            assert self._U.shape == (self._N0, self._N0)
            assert np.amin(self._U) >= 0
        else:
            self._U = np.zeros((self._N0, self._N0), dtype=np.int64)
            
        if AC is not None:  # preset IC
            self._AC = AC
            assert self._AC.ndim == 1
            assert len(self._AC) == self._N0
            assert np.amin(self._AC) >= 0
        else:  # create AC
            self._AC = np.zeros(self._N0, dtype=np.int64)
            
        if states is not None:
            self._states = states
            assert np.amin(self._states) >= 0 
            assert np.amax(self._states) < self._N0
        else:
            self._states = []
        


    def update_transition(self, state_i: np.int64, state_j: np.int64):
        """
        1 is added to state_i in the internal update transition_matrix to
        record and indicate a state change to state_i.
        """
        assert state_i >= 0 and state_i < self._N0
        assert state_j >= 0 and state_j < self._N0
        if len(self._states):
            if state_i != self._states[-1]:
                raise ValueError(f'Jump state transition! From state {state_i} != previous state {self._states[-1]}')
            else:
                self._states.append(state_j)
        else:
            self._states.append(state_i)
            self._states.append(state_j)
                

        # Records a state transition from state i to state j
        self._U[state_i, state_j] += 1
        # Records a state visit in state i
        self._AC[state_i] += 1       
		


    def get_state_updates(self):
        return self._AC
    

    def get_known_states(self):
        return (self._AC != 0).nonzero()[0]
    

    def get_full_transition_matrix(self):
        N0 = self._N0
        U = self._U
        P0 = self._P0

        # Create resulting transition matrix
        F = np.zeros((N0, N0), dtype=np.float64)
        for row in range(N0):
            if self._AC[row]:
                divisor = 1 + self._AC[row]
                prev = P0[row, ] / divisor
                update = U[row, ] / divisor
                F[row, ] = prev + update
            else:
                F[row, ] = P0[row, ]
            assert np.isclose(1, np.sum(F[row, ]))
            assert np.amin(F[row,]) >= 0
        return F, self._U, self._AC, self._states  

    def get_actual_transition_matrix(self):
        F, U, AC, states = self.get_full_transition_matrix()
        ls = states[-1]
        tc = self._IC + AC
        # get the indexes of the known states
        ks_idx = list((tc > 1).nonzero()[0])
        if 0 == len(ks_idx):
            raise ValueError('No state transition updated!')
        if ls not in ks_idx:
            ks_idx.append(ls) # this is because AC is updated after leaving state i, not entering
            ks_idx = np.sort(ks_idx)
            print(f'{Fore.YELLOW}Warning!!!{Style.RESET_ALL} No transition from state {ls} was reported. This will distort stable distribution!')
        P = F[np.ix_(ks_idx, ks_idx)]
        assert np.amin(P) >= 0
        if not np.isclose(np.sum(P, axis=1), np.ones(P.shape[0], dtype=np.float64)).all():
            raise ValueError('Improper transition matrix, not all row sum to 1. May be the state transition update is too few')
		# the returned count could be 1 because AC is incremented after leaving state i, this is warned above! 	
        return P, dict(zip(ks_idx, tc[ks_idx]))

    def get_encountered_rate(self):
        tc = (self._IC + self._AC)
        sum = np.sum(tc)
        erv = np.zeros(self._N0, dtype=np.float64)
        for i, cnt in enumerate(tc):
            erv[i] = cnt / sum	
        return erv, tc
		
    def prune(self, threshold_ratio = 0.0001):
        """
        1 -> 2 -> 3 -> 4 -> 5
        """
        tc = (self._IC + self._AC)
        sum = np.sum(tc)
        pruned_states = []
        zeros = np.zeros(self._N0)
        for i, cnt in enumerate(tc):
            assert cnt >= 1
            if cnt / sum <= threshold_ratio and cnt > 1:
                pruned_states.append(i)
                states = self._states
                Ns = len(states)
                if Ns < 2:
                    raise ValueError('Too few states')
                
                # loop thru states, update counts on U 
                for k, state in enumerate(states):
                    if (state==i): 
                        # found state to be pruned
                        if 0 == k: # first
                            self._U[state, states[1]] = self._U[state, states[1]] - 1
                            assert self._U[state, states[1]] >= 0
                        elif Ns-1 == k: # last
                            self._U[states[k-1], state] = self._U[states[k-1], state] - 1
                            assert self._U[states[k-1], state] >= 0
                        else: # in the middle
                            self._U[states[k-1], state] = self._U[states[k-1], state] - 1
                            if self._U[states[k-1], state] < 0:
                                assert states[k-1] == state
                                self._U[states[k-1], state] = 0
                            assert self._U[states[k-1], state] >= 0
                            self._U[state, states[k+1]] = self._U[state, states[k+1]] - 1
                            if self._U[state, states[k+1]] < 0:
                                assert state == states[k+1]
                                self._U[state, states[k+1]] = 0
                            assert self._U[state, states[k+1]] >= 0
                            if (states[k-1] != i):
                                l = k + 1
                                while states[l] == i and l < Ns:
                                    l = l + 1
                                self._U[states[k-1], states[l]] = self._U[states[k-1], states[l]] + 1
                assert 0 == np.amin(self._U[i,])
                            
                # remove state i in the states
                new_states = []
                for state in states:
                    if state != i:
                        new_states.append(state)
                self._states = new_states
                
        if len(pruned_states):
            print(f'pruned_states={pruned_states}')
            # Update AC counts
            self._AC = np.sum(self._U, axis=1)
            
		
    def get_stable_distribution_brute(self, power=100):
        """
        The matrix multiplied by itself many times will result in every row 
        of the matrix being equal to the stable distribution.
        """
        P, states_dict = self.get_actual_transition_matrix()

        t = P

        for _ in range(power):
            t = np.matmul(t, P)
        sd = np.mean(t, axis=0)
    
        assert np.isclose(np.sum(sd), 1)
        return sd, list(states_dict.keys())


    def get_stable_distribution(self):
        """
        solve for the stable distribution
        the stable distribution must be a solution to 
        (I-P')x=0
        at the same time, all x's elements must sum up to one
        so we add another equation to the above forming
        |      |  |0|
        |I - P'|x=|.|
        |      |  |0|
        |  e'  |  |1|
        or in shortform
        Ax=b
        This is an overdetermined system of linear equations, 
        there are n+1 equations but only n unknowns
        It can be solved by the QR decompositions. 
        Decompose A=QR, so Ax=b becomes
        QRx=b, so Rx=Q'b. 
		Solve this system of equations. 
		Here R is nxn
        """
        # Form the I-P' matrix
        P, states_dict = self.get_actual_transition_matrix()

        n = P.shape[0]
        m = np.identity(n) - P.T
        m = np.vstack((m, np.ones(n)))
        b = np.zeros(n) 
        b = np.append(b, 1)
        b = np.reshape(b, (n+1, 1))
        Q, R = np.linalg.qr(m, mode='reduced')
        #print(Q.shape, R.shape)
        #print(np.matmul(Q.T, Q)
        Qb = np.matmul(Q.T, b)
        stable_distribution = np.linalg.solve(R, Qb)
        sd = stable_distribution.squeeze()
        nn = 0
        for i, d in enumerate(sd):
            if d < 0:
                nn = nn + 1
                assert abs(d) < 1e-15
                sd[i] = 0
        if nn: # renormalize
            sd = sd / np.sum(sd)
        assert np.isclose(np.sum(sd), 1)
        return sd, list(states_dict.keys())


if __name__ == '__main__':
    def test(Nd): 
        assert Nd > 10
        offset = np.random.randint(0, Nd - 10)
        Total_Nd = Nd*2 + offset
        
        print(f'Total seq = {Total_Nd}, break point at {Nd - offset}')
        
        from deepdiff import DeepDiff
        
        Ns = 500    # actual number of states
        N0 = 520    # possible number of states
        
        # create state sequence
        states = np.random.randint(0, Ns, Total_Nd)
        #states = np.loadtxt('states.txt').astype(int)
        
        # create continuous session
        mc = MCTransitionMatrix(N0)
        
        # update continuous session
        for i in range(Total_Nd-1):
            mc.update_transition(states[i], states[i+1])
            
        # retrieve continuous session results  
        P, dict = mc.get_actual_transition_matrix()
        sd, s = mc.get_stable_distribution()
        
        print(f'States seens so far = {len(mc.get_known_states())}')
    
        # create 1st half persisted session
        mc1 = MCTransitionMatrix(N0)
        # update 1st half persisted session
        for i in range(Nd - offset):
            mc1.update_transition(states[i], states[i+1])
        
        # get 1st half persisted session results
        F, U, AC, states_so_far = mc1.get_full_transition_matrix()
        
        # create 2nd half persisted session using the 1st half results
        mc2 = MCTransitionMatrix(N0, U=U, AC=AC, states=states_so_far)
        
        # update 2nd half persisted session
        for i in range(Nd - offset, Total_Nd - 1):
            mc2.update_transition(states[i], states[i+1])
            
        # get 2nd half persisted session results 
        P2, dict2 = mc2.get_actual_transition_matrix()
        sd2, s2 = mc2.get_stable_distribution()
        
        tol = 1e-16  # very tight tolorance 
        
        # verify continuous session is the same as persisted sessions
        assert np.allclose(P, P2, rtol=tol)
        assert 0==len(DeepDiff(dict, dict2))
        assert np.allclose(sd, sd2, rtol=tol)
        assert np.array_equal(s, s2)
        
        # get encountered rate
        er, tc = mc.get_encountered_rate()
        er_set = sorted(set(er))
        
        # the last N0-Ns states never happen, so prune them will not change anything
        mc.prune(threshold_ratio=er_set[0])
        P3, dict3 = mc.get_actual_transition_matrix()
        sd3, s3 = mc.get_stable_distribution()
        
        # verify
        assert np.allclose(P, P3, rtol=tol)
        assert 0==len(DeepDiff(dict, dict3))
        assert np.allclose(sd, sd3, rtol=tol)
        assert np.array_equal(s, s3)
        
        # now do some real prunes 
        mc.prune(threshold_ratio=er_set[2])
        P4, dict4 = mc.get_actual_transition_matrix()
        sd4, s4 = mc.get_stable_distribution()
        
        print(f'Continuous session prune : P.shape={P.shape}, P4.shape={P4.shape}')
       
        # same should be for persisted sessions
        mc2.prune(threshold_ratio=er_set[2])
        P5, dict5 = mc2.get_actual_transition_matrix()
        sd5, s5 = mc2.get_stable_distribution()
        
        print(f'Persisted session prune : P.shape={P.shape}, P5.shape={P5.shape}')
        
        # Both session results should be the same
        assert np.allclose(P4, P5, rtol=tol)
        assert 0==len(DeepDiff(dict4, dict5))
        assert np.allclose(sd4, sd5, rtol=tol)
        assert np.array_equal(s4, s5)
        
        print(f'States after prune = {len(mc.get_known_states())}')

        print(f'=========== {Fore.GREEN}All tests passed!{Style.RESET_ALL} ============')
    
    Nd = 1000
    for _ in range(1000):
        test(Nd)
        Nd = Nd + np.random.randint(1, 10)
   
# %%