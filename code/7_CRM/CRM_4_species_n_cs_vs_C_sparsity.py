import numpy as np
from scipy.integrate import solve_ivp
import argparse
import json
import time
from pathlib import Path
from joblib import Parallel, delayed
import scipy.stats as st

from CRM import CRM
from CRM_utils import make_D, numerical_error, has_converged





def run_simulation(n_cs, C_sparsity, iterations, leakage, initial_c_conc, initial_abundance,
                   time, dilution_rate, n_species, K_mean, K_std = 1):
    
    final_abundance_matrix = np.zeros((iterations, n_species))
    
    for k in range(iterations):
        R0 = np.zeros(n_cs)
        R0[0] = initial_c_conc
        l = leakage * np.ones((n_species, n_cs))
        N0 = np.ones(n_species) * initial_abundance
        
        C = np.zeros((n_species, n_cs))
        while C[:, 0].max()<2*dilution_rate:
            C = np.random.lognormal(0, 1, (n_species, n_cs))
            C0 = (np.random.uniform(0,1, size = (n_species, n_cs))>C_sparsity).astype(int)
            C = C*C0
            # C = (C.T/C.sum(axis=1)).T # The existing data is with this line on - toggle off!
            C = C/C.max()


        D  = np.random.uniform(0, 1, size = (n_species, n_cs, n_cs))
        for i in range(n_cs):
            D[:,i,i] = 0
        # np.fill_diagonal(D, 0)
        for i in range(n_species):
            D[i,:,:] = D[i,:,:]/D[i, :,:].sum(axis = 0)
        # D /= D.sum(axis=2, keepdims=True)
        D = np.transpose(D, (0, 2, 1))

        
        K = np.random.lognormal(K_mean, K_std, (n_species, n_cs))
        c = CRM(n_species, n_cs, C, D=D, dilution_rate=dilution_rate, l=l, K=K)
        
        try:
            sol = c.run(time, N0, R0, dt=0.1, method='BDF')
        except ValueError:
            pass
        else:
            if has_converged(c.N) and not numerical_error(c.N):
                final_abundance_matrix[k, :] = c.N[-1, :]

    return final_abundance_matrix

def run_simulation_transfer(n_cs, C_sparsity, iterations, leakage, initial_c_conc, initial_abundance,
                   time, dilution_rate, n_species, K_mean, K_std = 1):
    
    final_abundance_matrix = np.zeros((iterations, n_species))
    
    for k in range(iterations):
        R0 = np.zeros(n_cs)
        R0[0] = initial_c_conc
        l = leakage * np.ones((n_species, n_cs))
        N0 = np.ones(n_species) * initial_abundance
        
        C = np.zeros((n_species, n_cs))
        while C[:, 0].max()<2*dilution_rate:
            C = st.gamma.rvs(1.2, 0.01, 0.16, size = (n_species, n_cs))
            C0 = (np.random.uniform(0,1, size = (n_species, n_cs))>C_sparsity).astype(int)
            C = C*C0
            # C = (C.T/C.sum(axis=1)).T # The existing data is with this line on - toggle off!
            C = C/C.max()


        D = st.lognorm.rvs(0.95, 2e-06, 0.05, size = (n_species, n_cs, n_cs))
        for i in range(n_cs):
            D[:,i,i] = 0
        # np.fill_diagonal(D, 0)
        for i in range(n_species):
            D[i,:,:] = D[i,:,:]/D[i, :,:].sum(axis = 0)
        # D /= D.sum(axis=2, keepdims=True)
        D = np.transpose(D, (0, 2, 1))

        
        K = np.random.lognormal(K_mean, K_std, (n_species, n_cs))
        c = CRM(n_species, n_cs, C, D=D, dilution_rate=dilution_rate, l=l, K=K)
        
        try:
            time = 24
            sol = c.run_transfers(time, N0,R0, dt = 1, transfer_dilution=100, n_transfers=30, method='BDF')
        except ValueError:
            pass
        else:
            if not numerical_error(c.N):
                final_abundance_matrix[k, :] = c.N[-1, :]

        return final_abundance_matrix

            

def run_parameter_swipe_parallel(n_cs_arr, C_sparsity_arr, iterations, leakage, initial_c_conc=10, 
                                  initial_abundance=1e-2, time=1000, dilution_rate=0.1, 
                                  n_species=4, K_mean=-1, K_std = 1, transfer = False):
    
    params_list = []
    for n_cs in n_cs_arr:
        for c_sp in C_sparsity_arr:
            params_list.append([n_cs, c_sp, iterations, leakage, initial_c_conc, initial_abundance,
                                time, dilution_rate, n_species, K_mean, K_std])
    
    num_jobs = len(params_list)
    if transfer:
        print('Run transfer')
        results = Parallel(n_jobs=-1)(delayed(run_simulation_transfer)(*params) for params in params_list)
    else:
        results = Parallel(n_jobs=-1)(delayed(run_simulation)(*params) for params in params_list)
    # results = Parallel(n_jobs=-1)(delayed(test)(params[0]) for params in params_list)

    
    # Reshape results into final abundance matrix
    final_abundance_matrix = np.array(results).reshape(len(n_cs_arr), len(C_sparsity_arr), iterations, n_species)
    
    return final_abundance_matrix


if __name__ == '__main__':
    t0 = time.time()
    parser = argparse.ArgumentParser(description = 'Parameter swipe')
    parser.add_argument("--cs_min", help = 'Minimal number of CS', default=2, type=int)
    parser.add_argument("--cs_max", help = 'Maximal number of CS', default=30, type=int)
    parser.add_argument("--N_cs", help = 'The number of different sizes of CS', default=29, type=int)
    parser.add_argument("--K_mean", help = 'Mean K', default = -1, type=float)
    parser.add_argument("--csp_min", help = 'Minimal C_sparsity', default = 0, type=float)
    parser.add_argument("--csp_max", help = 'Maximal C_sparsity', default = 1, type=float)
    parser.add_argument("--N_K_std", help = 'Number of different C sparsity values', default = 21, type=int)
    parser.add_argument("--iterations", help = 'Number of iterations per parameter combination', default = 10, type=int)
    parser.add_argument("--leakage", help = "Leakage fraction", default=0.1, type=float)
    parser.add_argument("--folder", help = 'Where to store data', default = './parameter_swipe_c_sparsity', type=str)
    parser.add_argument("--transfer", help = 'Run serial transfers', default = False, type=bool)


    
    args = parser.parse_args()
    args.filename = 'CRM_4_species_n_vs_C_sparsity.py'
    timestr = time.strftime("%Y%m%d-%H%M%S")

    n_cs_arr = np.linspace(args.cs_min, args.cs_max, num=args.N_cs, dtype=int)
    C_sparsity_arr = np.linspace(args.csp_min, args.csp_max, num = args.N_K_std)

    final_abundance_matrix = run_parameter_swipe_parallel(n_cs_arr, C_sparsity_arr, args.iterations, args.leakage, K_mean=args.K_mean, transfer=args.transfer)

    # Save final abundance amtrix and parameters
    Path(args.folder).mkdir(exist_ok=True, parents = True)

    fn_1 = Path(args.folder) / f'{timestr}_final_abundance.npz'
    fn_2 = Path(args.folder) / f'{timestr}_args.txt'

    with open(fn_1, 'wb') as f:
        np.savez(f, final_abundance_matrix)
    

    with open(fn_2, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    total_seconds = time.time()-t0
    total_minutes = total_seconds/60
    print(f"Total minutes: {total_minutes:.2f}")
    print(n_cs_arr)
    print(C_sparsity_arr)

