#!/usr/bin/env python3

import numpy as np
from scipy.integrate import solve_ivp


"""
Author: Snorre Sulheim
Email: snorre.sulheim@unil.ch


Simple implementation of the consumer-resource model similar to implementations used in:
- Goldford et al., Science 2018
- Pacheco et al., Nat. Comms 2021
- Marsland et al., PLoS One, 2020
Includes:
 - leakage
 - Cross-feeding

Does not include:
 - toxicity 
 - auxotrophy

Todo:
- Add a class for sampling random species / parameters

"""
epsilon = 1e-6
N_min = 1e-7
R_min = 1e-9
min_Jin = 1e-7
max_iterations = 10000

class IterationLimit:
    def __init__(self, max_calls):
        self.calls = 0
        self.max_calls = max_calls
    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls > self.max_calls:
            raise RuntimeError("Maximum iterations exceeded")
        
iteration_limit = IterationLimit(max_iterations)


def steady_state_event(t, y, ns, nr, C, K, g, w, l, m, D, dilution_rate, R_in):
    # y[:ns] are populations
    N = y[:ns]
    R = y[ns:]
    Jin = calc_Jin(R, C, K)
    dN_dt = calc_dN_dt(N, g, m, Jin, w, l, ns, nr, dilution_rate)
    # Trigger event when max absolute change is below threshold
    return np.max(np.abs(dN_dt)) - epsilon  # Set your threshold here

steady_state_event.terminal = True
steady_state_event.direction = -1

class CRM(object):
    """
    Consumer Resource Model (CRM) class for simulating consumer-resource dynamics.

    # g
    FBA E. coli on glucose suggests that the yield on glucose is 0.087 g biomass per mmol glucose.
    However, according to FBA (on glucose) 38% of carbon is released as CO2. Roughly 10% is lost as other byproducts.
    Thus, roughly the yield is roughly 0.167 g/mol when accounting for byproducts. If not CO2 is included maybe a yield og 0.1 g/mmol is more appropriate.
    """
    def __init__(self, N_species, N_resources, C, K = None, g = 0.1, 
                 w = None,
                 l = None,
                   m = None,
                     D = None,
                  dilution_rate = 0, R_in = None, transfer_time = None, 
                  transfer_dilution = None,
                  rtol = 1e-9, atol = 1e-9):
        self.N_resources = N_resources
        self.N_species = N_species
        self.R_in = R_in
        self.rtol = rtol
        self.atol = atol
        
        self._set_and_check_params(C, K, g, w, l, m, D, dilution_rate)

    def _set_and_check_params(self, C, K, g, w, l, m, D, dilution_rate):
        self.C = np.array(C)
        assert self.C.shape == (self.N_species, self.N_resources)
        
        if isinstance(K, (int, float)):
            self.K = K * np.ones((self.N_species, self.N_resources))
        elif isinstance(K, (list, np.ndarray)):
            K = np.array(K)
            if len(K.shape) == 1:
                assert len(K) == self.N_resources
                self.K = np.repeat(K[np.newaxis, :], self.N_species, axis=0)
            else:
                assert K.shape == (self.N_species, self.N_resources)
                self.K = K
        else:
            self.K = np.ones((self.N_species, self.N_resources))

        if g is None:
            self.g = 1
        elif isinstance(g, list):
            self.g = np.array(g)
        else:
            self.g = g

        if w is None:
            self.w = 1
        elif isinstance(w, list):
            self.w = np.array(w)
        else:
            self.w = w

        if isinstance(l, (int, float)):
            self.l = l * np.ones(self.N_species)
        elif isinstance(l, (list, np.ndarray)):
            self.l = np.array(l)
            assert self.l.size == self.N_species
        else:
            self.l = np.zeros(self.N_species)
            

        if isinstance(m, (int, float)):
            self.m = m * np.ones(self.N_species)
        elif isinstance(m, (list, np.ndarray)):
            self.m = np.array(m)
            assert self.m.size == self.N_species
        else:
            self.m = np.zeros(self.N_species)

        if not isinstance(D, (list, np.ndarray)):
            self.D = np.zeros((self.N_resources, self.N_resources))
            if np.sum(self.l)!= 0:
                print("D must be specified if leakage is != 0")
                raise IOError
        else:
            self.D = np.array(D)
            if len(self.D.shape) == 2:
                # 2-dim matrix-> 3D
                self.D = np.repeat(self.D[np.newaxis, :, :], self.N_species, axis=0)

            if not (np.abs(np.sum(self.D, axis = 2) - 1) < epsilon).all():
                print("D rows must sum to 1")
                print(np.abs(np.sum(self.D, axis = 1) - 1))
                raise IOError
        
        self.dilution_rate = dilution_rate


    def run_transfers(self, t_transfer, N0, R0, dt = 0.1, n_transfers = 1, 
                      transfer_dilution = 100, method = 'RK45'):
        self.dilution_rate = 0
        ns, nr = self.N_species, self.N_resources
        # if not self.R_in:
        self.R_in = R0

        N_list = []
        R_list = []
        for i in range(n_transfers):
            if i == 0:
                y0 = np.concatenate((N0, R0))
            else:
                Ni = arr[:self.N_species, -1]/transfer_dilution
                y0 = np.concatenate((Ni, R0))
            
            sol = solve_ivp(CRM_fun_with_limit, [0, t_transfer], y0, args = (ns, nr, self.C, self.K, self.g, 
                                                    self.w, self.l, self.m, self.D,
                                                    self.dilution_rate, self.R_in), dense_output=True, method=method)
            # print("N solver steps", len(sol.t))

            t = np.arange(0, t_transfer, dt)
            arr = sol.sol(t)
            N_list.append(arr[:self.N_species].T)
            R_list.append(arr[self.N_species:].T)
        self.N = np.concatenate(N_list)
        self.R = np.concatenate(R_list)
        return sol

        



    def run(self, t_max, N0, R0, dt = 0.1, method = 'RK45', max_calls = None):
        y0 = np.concatenate((N0, R0))
        ns, nr = self.N_species, self.N_resources
        if not self.R_in:
            self.R_in = R0
        
        iteration_limit.calls = 0  # Reset the iteration limit counter
        if max_calls is None:
            iteration_limit.max_calls = int(max(1e4, int(np.round(100 * ns * nr**2,0)))) # Set the maximum number of iterations based on system size
        else:
            iteration_limit.max_calls = max_calls
        if self.dilution_rate > 0:
            sol = solve_ivp(CRM_fun_with_limit, [0, t_max], y0, args = (ns, nr, self.C, self.K, self.g, 
                                                        self.w, self.l, self.m, self.D,
                                                        self.dilution_rate, self.R_in), 
                            dense_output=True, method = method,
                            first_step=1e-2,
                            min_step = 1e-6,
                            rtol = self.rtol, atol = self.atol)
        if sol.success is False:
            print(sol.message)
        # print(iteration_limit.calls, iteration_limit.max_calls)
        t = np.arange(0, t_max, dt)
        arr = sol.sol(t)
        self.N = arr[:self.N_species].T
        self.R = arr[self.N_species:].T
        return sol


def calc_Jin(R, C, K):
    # R is a vector of length nr 
    # K is a matrix of length ns x nr

    if not np.isfinite(C).all() or (C < 0).any():
        print("C has negative or non-finite values")
        raise ValueError
    if not np.isfinite(K).all() or (K < 0).any():        
        print("K has negative or non-finite values")
        raise ValueError
    
    u = R[None, :]/(R[None, :]+K)
    Jin = u * C 
    return Jin

def calc_dN_dt_loop(N, g, m, Jin, w, l, ns, nr, dilution_rate = 0):
    dN_dt = np.zeros(ns)
    for i in range(ns):
        dN_dt[i] = N[i]*g[i]*(np.sum(w*(1-l[i])*Jin[i,:])-m[i]) - N[i]*dilution_rate
    return dN_dt

def calc_dN_dt(N, g, m, Jin, w, l, ns, nr, dilution_rate = 0):
    # Vectorized version
    growth = g * (np.sum(w * (1 - l[:, None]) * Jin, axis=1) - m)
    dN_dt = N * (growth - dilution_rate)
    return dN_dt

def calc_dN_dt_simple(N, g, Jin, l,  dilution_rate = 0):
    # Vectorized version
    growth = g * np.sum((1 - l[:, None]) * Jin, axis=1)
    dN_dt = N * (growth - dilution_rate)
    return dN_dt
    
def calc_dN_dt_simple(N, g, Jin, l,  dilution_rate = 0):
    # Vectorized version
    growth = g * np.sum((1 - l[:, None]) * Jin, axis=1)
    dN_dt = N * (growth - dilution_rate)
    return dN_dt

def calc_dN_dt_simple_batch(N, g,Jin, l):
    # Vectorized version
    growth = g * np.sum((1 - l[:, None]) * Jin, axis=1)
    dN_dt = N * growth
    return dN_dt

def calc_dR_dt_no_crossfeeding(N, Jin, ns, nr):
    # ns, nr = Jin.shape
    dR_dt = np.zeros(nr)
    for j in range(nr):
        dR_dt[j]= -np.sum(N*Jin[:, j])
    return dR_dt

def calc_dR_dt_simple(N, Jin, D, l, dilution_rate = 0, R_in = 0, R = 0):
    # Jin: (ns, nr), N: (ns,), D: (ns, nr, nr), l: (ns,)
    Rup_matrix = Jin * N[:, np.newaxis]  # shape: (ns, nr)
    Rl = Rup_matrix * l[:, np.newaxis]   # shape: (ns, nr)
    leakage = D * Rl[:, :, np.newaxis]   # shape: (ns, nr, nr)
    leakage_sum = leakage.sum(axis=(0, 1))  # sum over species and consumed resource
    dR_dt = -np.sum(Rup_matrix, axis=0) + leakage_sum

    if dilution_rate > 0:
        dR_dt += (R_in - R) * dilution_rate
    return dR_dt

def calc_dR_dt_simple_batch(N, Jin, D, l):
    # Jin: (ns, nr), N: (ns,), D: (ns, nr, nr), l: (ns,)
    Rup_matrix = Jin * N[:, np.newaxis]  # shape: (ns, nr)
    Rl = Rup_matrix * l[:, np.newaxis]   # shape: (ns, nr)
    leakage = D * Rl[:, :, np.newaxis]   # shape: (ns, nr, nr)
    leakage_sum = leakage.sum(axis=(0, 1))  # sum over species and consumed resource
    dR_dt = -np.sum(Rup_matrix, axis=0) + leakage_sum
    return dR_dt

def calc_dR_dt(N, Jin, D, w, l, ns, nr, dilution_rate = 0, R_in = 0, R = 0):
    # Vectorized leakage calculation
    # D: (ns, nr, nr), w: (nr,), l: (ns,), N: (ns,), Jin: (ns, nr)
    if isinstance(w, (np.ndarray)):
        w_ratio = w[np.newaxis, :, np.newaxis] / w[np.newaxis, np.newaxis, :]  # shape: (1, nr, nr)
    else:
        w_ratio = 1
        
    lN = (l * N)[:, np.newaxis, np.newaxis]  # shape: (ns, 1, 1)
    Jin_expanded = Jin[:, :, np.newaxis]     # shape: (ns, nr, 1)
    leakage = D * w_ratio * lN * Jin_expanded  # shape: (ns, nr, nr)
    leakage_sum = leakage.sum(axis=(0, 1))     # sum over species and consumed resource

    dR_dt = -np.sum(N[:, np.newaxis] * Jin, axis=0) + leakage_sum

    if dilution_rate > 0:
        dR_dt += (R_in - R) * dilution_rate
    return dR_dt

def calc_dR_dt_loop(N, Jin, D, w, l, ns, nr, dilution_rate = 0, R_in = 0, R = 0):
    dR_dt = np.zeros(nr)
    for j in range(nr):
        leakage = np.zeros((ns, nr))
        for i in range(ns):
            for k in range(nr):
                leakage[i, k] = D[i, k, j]*(w[k]/w[j])*l[i]*N[i]*Jin[i,k] # D[i, j, k] if respource partitioning is species specific, check 
        dR_dt[j]= -np.sum(N*Jin[:, j])+ np.sum(leakage)
        
    if dilution_rate > 0:
        dR_dt += (R_in-R)*dilution_rate
    return dR_dt



def CRM_fun_with_limit(t, y, ns, nr, C, K, g, w, l, m, D, dilution_rate = 0, R_in=0):
    iteration_limit()
    return CRM_fun(t, y, ns, nr, C, K, g, l, D, dilution_rate, R_in)

def CRM_fun(t, y, ns, nr, C, K, g, l, D, dilution_rate = 0, R_in=0):
    N = y[:ns]
    R = y[ns:]
    N[N<N_min] = 0
    R[R<R_min] = 0
    R[R==np.inf] = 0

    if np.sum(N) < N_min:
        return np.zeros_like(y)

    if not np.isfinite(R).all() or (R < 0).any():
        print(R)
        print("R has negative or non-finite values")
        raise ValueError


    Jin = calc_Jin(R, C, K)
    # Jin[Jin < min_Jin] = 0

    
    # dN_dt = calc_dN_dt(N, g, m, Jin, w, l, ns, nr, dilution_rate)
    dN_dt = calc_dN_dt_simple(N, g, Jin, l, dilution_rate)
    # dR_dt = calc_dR_dt(N, Jin, D, w, l, ns, nr, dilution_rate, R_in, R)
    dR_dt = calc_dR_dt_simple(N, Jin, D, l, dilution_rate, R_in, R)
    
    dN_dt[N<N_min] = 0

    return np.concatenate([dN_dt, dR_dt])


def CRM_fun_with_limit_batch(t, y, ns, nr, C, K, g, w, l, m, D, dilution_rate = 0, R_in=0):
    iteration_limit()
    return CRM_fun_batch(t, y, ns, nr, C, K, g, l, D, dilution_rate, R_in)


def CRM_fun_batch(t, y, ns, nr, C, K, g, l, D, dilution_rate = 0, R_in=0):
    N = y[:ns]
    R = y[ns:]
    N[N<N_min] = 0
    R[R<R_min] = 0
    R[R==np.inf] = 0


    if np.sum(N) < N_min:
        return np.zeros_like(y)

    if not np.isfinite(R).all() or (R < 0).any():
        print(R)
        print("R has negative or non-finite values")
        raise ValueError


    Jin = calc_Jin(R, C, K)
    # Jin[Jin < min_Jin] = 0

    
    # dN_dt = calc_dN_dt(N, g, m, Jin, w, l, ns, nr, dilution_rate)
    dN_dt = calc_dN_dt_simple_batch(N, g, Jin, l)
    # dR_dt = calc_dR_dt(N, Jin, D, w, l, ns, nr, dilution_rate, R_in, R)
    dR_dt = calc_dR_dt_simple_batch(N, Jin, D, l)
    
    # dN_dt[N<N_min] = 0

    # print('N:, ', dN_dt, N)
    # print('R:, ', dR_dt, R)
    # print(dR_dt)
    return np.concatenate([dN_dt, dR_dt])




if __name__ == "__main__":
    pass

