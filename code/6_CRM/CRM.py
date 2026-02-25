#!/usr/bin/env python3

import numpy as np
from scipy.integrate import solve_ivp, RK45, RK23, DOP853, Radau, BDF, LSODA


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
                  rtol = 1e-8, atol = 1e-12):
        self.N_resources = N_resources
        self.N_species = N_species
        self.R_in = R_in
        self.rtol = rtol
        self.atol = atol
        self.auxo_arr = None
        
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

        



    def run(self, t_max, N0, R0, dt = 0.1, method = 'RK45', max_calls = None, adaptive_tolerance=True):
        """Run the CRM simulation with robust integration settings.
        
        Parameters:
        -----------
        adaptive_tolerance : bool
            If True, automatically adjust tolerances based on system scale
        """
        y0 = np.concatenate((N0, R0))
        ns, nr = self.N_species, self.N_resources
        if not self.R_in:
            self.R_in = R0
        
        # Validate and clean initial conditions
        y0 = np.maximum(y0, 0.0)  # Ensure non-negative
        if not np.isfinite(y0).all():
            print("Warning: Non-finite initial conditions detected, cleaning...")
            y0 = np.nan_to_num(y0, nan=N_min, posinf=1e6, neginf=0.0)
        
        iteration_limit.calls = 0  # Reset the iteration limit counter
        if max_calls is None:
            iteration_limit.max_calls = int(max(1e4, int(np.round(100 * ns * nr**2,0)))) # Set the maximum number of iterations based on system size
        else:
            iteration_limit.max_calls = max_calls
        
        # Adaptive tolerance based on system scale
        rtol = self.rtol
        atol = self.atol
        if adaptive_tolerance:
            # Scale tolerances based on initial conditions
            typical_scale = np.median(y0[y0 > 0]) if np.any(y0 > 0) else 1.0
            atol = max(atol, typical_scale * 1e-9)
        
        # Choose appropriate solver settings
        if method in ['Radau', 'BDF', 'LSODA']:
            # Stiff solvers - better for CRM systems
            max_step = t_max / 10.0
            first_step = min(1e-3, t_max / 1000.0)
        else:
            # Non-stiff solvers
            max_step = t_max / 100.0
            first_step = min(1e-2, t_max / 1000.0)
        
        sol = solve_ivp(CRM_fun_with_limit, [0, t_max], y0, args = (ns, nr, self.C, self.K, self.g, 
                                                    self.w, self.l, self.m, self.D,
                                                    self.dilution_rate, self.R_in, self.auxo_arr), 
                        dense_output=True, method = method,
                        max_step = max_step,
                        first_step = first_step,
                        rtol = rtol, atol = atol)

        if sol.success is False:
            print(f"Integration failed: {sol.message}")
            print(f"Last time reached: {sol.t[-1]:.2f} / {t_max}")
            print(f"Function evaluations: {iteration_limit.calls}")
        
        # Safely extract solution
        try:
            t = np.arange(0, min(t_max, sol.t[-1]) + dt, dt)
            t = t[t <= sol.t[-1]]  # Clip to actual solution range
            arr = sol.sol(t)
            self.N = arr[:self.N_species].T
            self.R = arr[self.N_species:].T
        except Exception as e:
            print(f"Warning: Error extracting solution: {e}")
            # Use raw solution points
            self.N = sol.y[:self.N_species, :].T
            self.R = sol.y[self.N_species:, :].T
        
        return sol

    def run_log(self, t_max, N0, R0, dt=0.1, method='RK45', max_calls=None, adaptive_tolerance=True):
        """Run CRM with log-transformed variables for better numerical stability.
        
        Transforms the ODE system to work in log-space: d(log y)/dt = (1/y) * dy/dt
        This helps when populations/resources span many orders of magnitude.
        
        Parameters same as run() method.
        """
        # Minimum values for log transformation
        log_min = np.log(N_min)
        
        # Transform initial conditions to log space
        N0_safe = np.maximum(N0, N_min)
        R0_safe = np.maximum(R0, R_min) 
        y0_log = np.concatenate([np.log(N0_safe), np.log(R0_safe)])
        
        ns, nr = self.N_species, self.N_resources
        if not self.R_in:
            self.R_in = R0
        
        # Validate
        if not np.isfinite(y0_log).all():
            print("Warning: Non-finite log initial conditions")
            y0_log = np.nan_to_num(y0_log, nan=log_min, posinf=10, neginf=log_min)
        
        iteration_limit.calls = 0
        iteration_limit.max_calls = max_calls
        
        # Adaptive tolerances
        rtol = self.rtol
        atol = self.atol
        if adaptive_tolerance:
            atol = max(atol, 1e-8)  # Log space is typically more stable
        
        # Solver settings
        
        max_step = t_max / 10.0
        first_step = min(1e-3, t_max / 1000.0)

        # Integrate in log space
        sol = solve_ivp(CRM_fun_log_with_limit, [0, t_max], y0_log, 
                       args=(ns, nr, self.C, self.K, self.g, self.w, self.l, self.m, self.D,
                             self.dilution_rate, self.R_in, self.auxo_arr),
                       dense_output=True, method=method,
                       max_step=max_step, first_step=first_step,
                       rtol=rtol, atol=atol)
        
        if sol.success is False:
            print(f"Integration failed: {sol.message}")
            print(f"Last time reached: {sol.t[-1]:.2f} / {t_max}")
        
        # Transform back from log space
        try:
            t = np.arange(0, min(t_max, sol.t[-1]) + dt, dt)
            t = t[t <= sol.t[-1]]
            arr_log = sol.sol(t)
            
            # Clip extreme log values before exponentiating
            arr_log = np.clip(arr_log, log_min, 20)  # exp(20) ~ 5e8
            
            arr = np.exp(arr_log)
            self.N = arr[:self.N_species].T
            self.R = arr[self.N_species:].T
        except Exception as e:
            print(f"Warning: Error extracting solution: {e}")
            arr_log = np.clip(sol.y, log_min, 20)
            arr = np.exp(arr_log)
            self.N = arr[:self.N_species, :].T
            self.R = arr[self.N_species:, :].T
        
        return sol

    def run_step(self, t_max, N0, R0, dt = 0.1, method = 'RK45', max_calls = None, max_step=None):
        """Alternative integration method using step-wise solver for more control.
        
        This method manually steps through the integration, which can be more robust
        for difficult problems and allows for better monitoring and control.
        
        Parameters:
        -----------
        t_max : float
            Maximum integration time
        N0 : array
            Initial species abundances
        R0 : array
            Initial resource concentrations
        dt : float
            Time step for storing solution
        method : str
            ODE solver method ('RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA')
        max_calls : int
            Maximum number of function evaluations
        max_step : float
            Maximum step size for the solver
        
        Returns:
        --------
        dict : Solution information with success status, message, and nfev
        """
        y0 = np.concatenate((N0, R0))
        ns, nr = self.N_species, self.N_resources
        if not self.R_in:
            self.R_in = R0
        
        iteration_limit.calls = 0
        iteration_limit.max_calls = max_calls
        
        # Get the appropriate solver class
        solver_map = {
            'RK45': RK45,
            'RK23': RK23, 
            'DOP853': DOP853,
            'Radau': Radau,
            'BDF': BDF,
            'LSODA': LSODA
        }
        
        if method not in solver_map:
            print(f"Unknown method {method}, defaulting to RK45")
            method = 'RK45'
        
        SolverClass = solver_map[method]
        
        # Create ODE function wrapper
        def fun(t, y):
            iteration_limit()
            return CRM_fun_with_limit(t, y, ns, nr, self.C, self.K, self.g,
                                     self.w, self.l, self.m, self.D,
                                     self.dilution_rate, self.R_in, self.auxo_arr)
        
        # Initialize solver
        if max_step is None:
            max_step = t_max / 10.0
        
        solver = SolverClass(fun, 0, y0, t_max, 
                            rtol=self.rtol, atol=self.atol,
                            # min_step = 1e-7,
                            max_step=max_step, first_step=1e-3)
        
        # Storage for solution
        t_list = [0]
        y_list = [y0]
        
        # Step through the integration
        try:
            while solver.status == 'running':
                # Check iteration limit
                if iteration_limit.calls > iteration_limit.max_calls:
                    print(f"Iteration limit reached: {iteration_limit.calls} calls")
                    break
                
                # Take a step
                message = solver.step()
                
                if solver.status == 'failed':
                    print(f"Solver failed: {message}")
                    break
                
                # Store solution
                t_list.append(solver.t)
                y_list.append(solver.y.copy())
                
                # Check if we've reached t_max
                if solver.t >= t_max:
                    break
        
        except Exception as e:
            print(f"Integration error: {e}")
            success = False
            message = str(e)
        else:
            success = solver.status == 'finished' or solver.t >= t_max
            message = solver.status
        
        # Convert to arrays
        t_array = np.array(t_list)
        y_array = np.array(y_list)
        
        # Interpolate to regular time grid
        t_grid = np.arange(0, min(t_max, t_array[-1]) + dt, dt)
        if t_grid[-1] > t_array[-1]:
            t_grid = t_grid[t_grid <= t_array[-1]]
        
        # Simple linear interpolation
        N_interp = np.zeros((len(t_grid), ns))
        R_interp = np.zeros((len(t_grid), nr))
        
        for i, t_val in enumerate(t_grid):
            idx = np.searchsorted(t_array, t_val)
            if idx == 0:
                N_interp[i] = y_array[0, :ns]
                R_interp[i] = y_array[0, ns:]
            elif idx >= len(t_array):
                N_interp[i] = y_array[-1, :ns]
                R_interp[i] = y_array[-1, ns:]
            else:
                # Linear interpolation
                t0, t1 = t_array[idx-1], t_array[idx]
                y0, y1 = y_array[idx-1], y_array[idx]
                alpha = (t_val - t0) / (t1 - t0) if t1 > t0 else 0
                y_interp = y0 + alpha * (y1 - y0)
                N_interp[i] = y_interp[:ns]
                R_interp[i] = y_interp[ns:]
        
        self.N = N_interp
        self.R = R_interp
        
        # Return solution info in a format similar to solve_ivp
        sol_info = {
            'success': success,
            'message': message,
            'nfev': iteration_limit.calls,
            't': t_grid,
            'y': np.vstack([N_interp.T, R_interp.T])
        }
        
        return type('Solution', (), sol_info)()


def calc_Jin(R, C, K):
    """Calculate uptake rates using Monod kinetics with numerical stability.
    
    Jin[i,j] = C[i,j] * R[j] / (R[j] + K[i,j])
    """
    # R is a vector of length nr 
    # K is a matrix of length ns x nr

    # Handle non-finite values in parameters
    if not np.isfinite(C).all():
        C = np.nan_to_num(C, nan=0.0, posinf=0, neginf=0.0)
    if not np.isfinite(K).all():
        K = np.nan_to_num(K, nan=1.0, posinf=1, neginf=1e-6)
    
    # Ensure non-negative
    C = np.maximum(C, 0.0)
    K = np.maximum(K, 1e-6)  # Prevent division by zero
    
    # Monod kinetics: u = R / (R + K)
    # Numerically stable formulation
    R_expanded = R[None, :]  # shape: (1, nr)
    denominator = R_expanded + K  # shape: (ns, nr)
    
    # Avoid division by zero
    denominator = np.maximum(denominator, 1e-6)
    
    u = R_expanded / denominator
    Jin = u * C
    
    # Clip extreme values
    Jin = np.clip(Jin, 0.0, 1e2)
    
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

def calc_mu_lim_loop(g, Jin, auxo_arr, l, ns, w, m, epsilon=1e-6):
    limited_growth_rates = np.zeros(ns)
    Jin_adjusted = Jin.copy()
    
    for i in range(ns):
        # Extract scalar values for this species
        g_i = g[i] if isinstance(g, np.ndarray) else g
        m_i = m[i] if isinstance(m, np.ndarray) else m
        
        # Check if this species is auxotrophic
        if (auxo_arr[i] > 0).sum() == 0:
            # Non-auxotrophic: use standard growth calculation
            if isinstance(w, np.ndarray):
                limited_growth_rates[i] = g_i * (np.sum(w * (1 - l[i]) * Jin[i]) - m_i)
            else:
                limited_growth_rates[i] = g_i * (np.sum((1 - l[i]) * Jin[i]) - m_i)
            continue

        growth_limits = []
        aux_idxs = np.where(auxo_arr[i])[0]
        aux_yields = auxo_arr[i, aux_idxs]
        
        # Calculate growth limits from auxotrophic requirements
        for (aux_idx, aux_yield) in zip(aux_idxs, aux_yields):
            aux_uptake = Jin[i, aux_idx]
            # Growth limited by this auxotroph
            growth_limits.append(aux_uptake * (1-l[i]) * aux_yield)
        
        # Calculate growth from all carbon sources, weighted by energy content
        if isinstance(w, np.ndarray):
            mu_cs = g_i * (np.sum(w * (1 - l[i]) * Jin[i]) - m_i)
        else:
            mu_cs = g_i * (np.sum((1 - l[i]) * Jin[i]) - m_i)
        growth_limits.append(mu_cs)

        # The actual growth rate is limited by the most limiting resource
        mu_lim_i = min(growth_limits)
        limited_growth_rates[i] = mu_lim_i
        
        # If carbon source uptake would lead to growth exceeding the limit,
        # scale down carbon source uptake proportionally
        if mu_cs > mu_lim_i + epsilon:
            cs_uptake_scale_i = ((mu_lim_i/g_i) + m_i) / ((mu_cs/g_i) + m_i)
            Jin_adjusted[i] *= cs_uptake_scale_i
        
        # If auxotroph uptake would lead to growth exceeding the limit,
        # scale down auxotroph uptake
        for aux_idx, gl in zip(aux_idxs, growth_limits[:-1]):
            if gl > mu_lim_i + epsilon:
                Jin_adjusted[i, aux_idx] = Jin[i, aux_idx] * mu_lim_i / gl

    return limited_growth_rates, Jin_adjusted

def calc_dN_dt_simple_auxo(N, limited_growth_rates, dilution_rate=0):
    """Calculate dN/dt using pre-calculated limited growth rates.
    
    For auxotrophic species, growth is already limited by the most
    constraining resource (auxotroph or carbon source).
    """
    dN_dt = N * (limited_growth_rates - dilution_rate)
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
    # leakage[i,k,j] = D[i,k,j] * (w[k]/w[j]) * l[i] * N[i] * Jin[i,k]
    if isinstance(w, np.ndarray):
        # w[k] is consumed (numerator), w[j] is produced (denominator)
        w_consumed = w[:, np.newaxis]  # shape: (nr, 1)
        w_produced = w[np.newaxis, :]  # shape: (1, nr)
        w_ratio = w_consumed / np.maximum(w_produced, 1e-10)  # shape: (nr, nr), avoid division by zero
        w_ratio = w_ratio[np.newaxis, :, :]  # shape: (1, nr, nr) for broadcasting
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



def CRM_fun_with_limit(t, y, ns, nr, C, K, g, w, l, m, D, dilution_rate = 0, R_in=0, auxo_arr = None):
    iteration_limit()
    if auxo_arr is None:
        return CRM_fun(t, y, ns, nr, C, K, g, l, D, dilution_rate, R_in)
    else:
        return CRM_fun_auxo(t, y, ns, nr, C, K, g, l, D, dilution_rate, R_in, auxo_arr=auxo_arr, w=w, m=m)


def CRM_fun_log_with_limit(t, y_log, ns, nr, C, K, g, w, l, m, D, dilution_rate=0, R_in=0, auxo_arr=None):
    """Log-transformed ODE function: d(log(y))/dt = (1/y) * dy/dt
    
    Supports auxotrophy constraints when auxo_arr is provided.
    """
    iteration_limit()
    
    # Transform back to linear space for calculation
    y_log = np.clip(y_log, np.log(N_min), 20)  # Prevent extreme values
    y = np.exp(y_log)
    
    N = y[:ns]
    R = y[ns:]
    
    # Ensure positivity and handle edge cases
    N = np.maximum(N, N_min)
    R = np.maximum(R, R_min)
    
    # Early exit for dead system
    if np.sum(N) < N_min:
        return np.zeros_like(y_log)
    
    # Calculate Jin (uptake rates)
    Jin = calc_Jin(R, C, K)
    Jin = np.minimum(Jin, 1e6)
    
    # Handle auxotrophy if specified
    if auxo_arr is not None:
        # Calculate growth limited by auxotrophic requirements
        limited_growth_rates, Jin_adjusted = calc_mu_lim_loop(
            g, Jin, auxo_arr, l, ns, w, m, epsilon=1e-6
        )
        # Use limited growth rates for biomass dynamics
        dN_dt = calc_dN_dt_simple_auxo(N, limited_growth_rates, dilution_rate)
        # Use adjusted Jin for resource consumption
        dR_dt = calc_dR_dt(N, Jin_adjusted, D, w, l, ns, nr, dilution_rate, R_in, R)
    else:
        # Standard calculation without auxotrophy
        dN_dt = calc_dN_dt(N, g, m, Jin, w, l, ns, nr, dilution_rate)
        dR_dt = calc_dR_dt(N, Jin, D, w, l, ns, nr, dilution_rate, R_in, R)

    # Transform to log-space derivatives: d(log(y))/dt = (1/y) * dy/dt
    # Prevent division by zero
    dlog_N_dt = dN_dt / np.maximum(N, N_min)
    dlog_R_dt = dR_dt / np.maximum(R, R_min)
    
    # Clip extreme derivatives
    dlog_N_dt = np.clip(dlog_N_dt, -100, 100)
    dlog_R_dt = np.clip(dlog_R_dt, -100, 100)
    
    # Handle populations going to zero: if dy/dt < 0 and y is small, set d(log(y))/dt to 0
    dlog_N_dt[N <= N_min] = np.minimum(dlog_N_dt[N <= N_min], 0.0)
    dlog_R_dt[R <= R_min] = np.minimum(dlog_R_dt[R <= R_min], 0.0)
    
    return np.concatenate([dlog_N_dt, dlog_R_dt])


def CRM_fun(t, y, ns, nr, C, K, g, l, D, dilution_rate = 0, R_in=0):
    N = y[:ns]
    R = y[ns:]
    
    # Robust clipping with tolerance
    N = np.maximum(N, 0.0)
    R = np.maximum(R, 0.0)
    
    # Early exit if system is dead
    if np.sum(N) < N_min:
        return np.zeros_like(y)
    
    # Handle non-finite values gracefully
    if not np.isfinite(R).all():
        R = np.nan_to_num(R, nan=0.0, posinf=1e6, neginf=0.0)
    if not np.isfinite(N).all():
        N = np.nan_to_num(N, nan=0.0, posinf=1e6, neginf=0.0)
    
    # Clip very small values to zero to avoid numerical issues
    N[N < N_min] = 0
    R[R < R_min] = 0

    Jin = calc_Jin(R, C, K)
    
    # Clip Jin to prevent extreme values
    Jin = np.minimum(Jin, 1e6)
    
    dN_dt = calc_dN_dt_simple(N, g, Jin, l, dilution_rate)
    dR_dt = calc_dR_dt_simple(N, Jin, D, l, dilution_rate, R_in, R)
    
    # Prevent populations from going negative by setting derivative to 0
    dN_dt[N < N_min] = np.maximum(dN_dt[N < N_min], 0.0)
    
    # Prevent resources from going negative
    dR_dt[R < R_min] = np.maximum(dR_dt[R < R_min], 0.0)
    
    # Clip extreme derivatives
    dN_dt = np.clip(dN_dt, -1e6, 1e6)
    dR_dt = np.clip(dR_dt, -1e6, 1e6)

    return np.concatenate([dN_dt, dR_dt])


def CRM_fun_auxo(t, y, ns, nr, C, K, g, l, D, dilution_rate = 0, R_in=0, auxo_arr = None, w=1, m=0):
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

    limited_growth_rates, Jin_adjusted = calc_mu_lim_loop(g, Jin, auxo_arr, l, ns, w=w, m=m, epsilon=1e-6)
    
    dN_dt = calc_dN_dt_simple_auxo(N, limited_growth_rates, dilution_rate)
    # dN_dt = calc_dN_dt(N, g, m, Jin, w, l, ns, nr, dilution_rate)
    # dN_dt = calc_dN_dt_simple(N, g, Jin, l, dilution_rate)
    # dR_dt = calc_dR_dt(N, Jin, D, w, l, ns, nr, dilution_rate, R_in, R)

    dR_dt = calc_dR_dt_simple(N, Jin_adjusted, D, l, dilution_rate, R_in, R)
    
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

