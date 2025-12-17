import numpy as np
import argparse
import json
import time
from pathlib import Path
from joblib import Parallel, delayed
import scipy.stats as st
from tqdm import tqdm
import pickle

from CRM import CRM
from CRM_utils import make_D, numerical_error, has_converged, make_C

# GLOBAL_PARAMS
# From 6A_estmate_paramtersmax uptake rates consumer preference distribution
# D_log10_dist = st.norm(loc=-3.48, scale = 1.35)  # Parameters for log10 of D distribution
C_dist = st.lognorm(0.91, scale=1.44, loc=0)
D_dist = st.uniform(0, 1)

def run_simulation(n_cs, K_std, iterations, leakage, initial_c_conc, initial_abundance,
                   time, dilution_rate, n_species, K_mean,
                   C_sparsity=0.1, 
                   dt = 0.1,
                   method='LSODA'):
    """
    Run a simulation with the specified parameters and return the final abundance matrix.
    If transfer is True, run the transfer simulation; otherwise, run the standard CRM simulation.
    """

    # Fixed arrays
    R0 = np.zeros(n_cs)
    R0[0] = initial_c_conc

    if n_cs == 1:
        # No need to have leakage if there is no byproducts
        l = np.zeros(n_species)
    else:
        l = leakage * np.ones(n_species)

    N0 = np.ones(n_species) * initial_abundance    

    runtime_errors = 0
    k = 0
    k_control = 0
    k_numerical_error = 0
    k_not_converged = 0
    k_not_successful = 0
    species_list = []
    # print('C_sparsity:', C_sparsity)
    # print('K_std:', K_std)
    # print('n_cs:', n_cs)

    while k < iterations:
        C = make_C(n_species, n_cs, C_dist, C_sparsity, g_yield = 0.1, min_ratio=2, dilution_rate=dilution_rate)
            
        D = make_D(n_species, n_cs, D_dist= D_dist)

        K = np.random.lognormal(K_mean, K_std, (n_species, n_cs))
        c = CRM(n_species, n_cs, C, D=D, dilution_rate=dilution_rate, l=l, K=K, atol=1e-9, rtol=1e-9, g=0.1)

        try:
            sol = c.run(time, N0, R0, dt=dt, method=method)
        except (RuntimeError):
            print("Runtime error")
            print(f"n_cs: {n_cs}, K_std: {K_std}, C_sparsity: {C_sparsity}")
            # print("C:", C)
            # print("D:", D)
            runtime_errors += 1
        else:
            if has_converged(c.N, tol = 1e-7) and not numerical_error(c.N) and sol.success:
                # final_abundance_matrix[k, :] = c.N[-1, :]
                k += 1
                species = {'C':C, 'D':D, 'K':K, 'N_final':c.N[-1, :], 'R_final':c.R[-1, :], 'K_std':K_std, 'n_cs':n_cs, 'C_sparsity':C_sparsity}
                species_list.append(species)
            else:
                if numerical_error(c.N):
                    k_numerical_error += 1
                if not has_converged(c.N):
                    k_not_converged += 1
                if not sol.success:
                    k_not_successful += 1
        k_control += 1
        if k_control > iterations * 10:
            print("Too many iterations without convergence, breaking loop")
            print(f"Runtime errors encountered: {runtime_errors}")
            print(f"Numerical errors encountered: {k_numerical_error}")
            print(f"Not converged: {k_not_converged}")
            print(f"Total iterations attempted: {k_control}")
            print(f"Successful iterations: {k}")
            break
    print(f'Runtime errors: {runtime_errors}, Numerical errors: {k_numerical_error}, Not converged: {k_not_converged}, Not successful: {k_not_successful}, Successful: {k} out of {k_control} attempts')
    return species_list

def run_parameter_swipe_parallel(n_cs_arr, K_std_arr, C_sparsity_arr, iterations, leakage, initial_c_conc=10, 
                                  initial_abundance=1e-3, time=2000, dilution_rate=0.1, 
                                  n_species=4, K_mean=-1, n_jobs=1):
    
    params_list = []
    for n_cs in n_cs_arr:
        for K_std in K_std_arr:
            for C_sparsity in C_sparsity_arr:
                params_list.append([n_cs, K_std, iterations, leakage, initial_c_conc, initial_abundance,
                                time, dilution_rate, n_species, K_mean, C_sparsity])
    
    num_jobs = len(params_list)

    print('Run chemostat CRM simulation')
    results = Parallel(n_jobs=n_jobs)(
        tqdm(
            (delayed(run_simulation)(*params) for params in params_list),
            total=len(params_list)
        )
    )
    
    print(f"Finished {num_jobs} jobs")
    # final_abundances = [result[0] for result in results]
    # species_data = [result[1] for result in results]
    # Reshape results into final abundance matrix
    # final_abundance_matrix = np.array(final_abundances).reshape(len(n_cs_arr), len(K_std_arr), len(C_sparsity_arr), iterations, n_species)
    
    # return final_abundance_matrix, species_data
    return results


if __name__ == '__main__':
    t0 = time.time()
    parser = argparse.ArgumentParser(description = 'Parameter swipe')
    parser.add_argument("--cs_min", help = 'Minimal number of CS', default=1, type=int)
    parser.add_argument("--cs_max", help = 'Maximal number of CS', default=41, type=int)
    parser.add_argument("--N_cs", help = 'The number of different sizes of CS', default=16, type=int)
    parser.add_argument("--K_mean", help = 'Mean K', default = -1, type=float)
    parser.add_argument("--K_std_min", help = 'Minimal K std', default = 0.1, type=float)
    parser.add_argument("--K_std_max", help = 'Maximal K std', default = 2, type=float)
    parser.add_argument("--N_K_std", help = 'Number of different K std values', default = 11, type=int)
    parser.add_argument("--iterations", help = 'Number of iterations per parameter combination', default = 50, type=int)
    parser.add_argument("--leakage", help = "Leakage fraction", default=0.1, type=float)
    parser.add_argument("--folder", help = 'Where to store data', default = 'simulation_results/n_cs_vs_K_parameter_swipe', type=str)
    # parser.add_argument("--transfer", help = 'Run serial transfers', default = False, type=bool)
    parser.add_argument("--csp_min", help = 'Minimal C_sparsity', default = 0, type=float)
    parser.add_argument("--csp_max", help = 'Maximal C_sparsity', default = 0.4, type=float)
    parser.add_argument("--N_Csp", help = 'Number of different C sparsity values', default = 3, type=int)
    
    args = parser.parse_args()
    timestr = time.strftime("%Y%m%d-%H%M%S")

    n_cs_arr = np.linspace(args.cs_min, args.cs_max, num=args.N_cs, dtype=int, endpoint = True)
    K_std_arr = np.linspace(args.K_std_min, args.K_std_max, num = args.N_K_std, endpoint = True)
    c_arr = np.linspace(args.csp_min, args.csp_max, num = args.N_Csp, endpoint = True)
    print(n_cs_arr)
    print(K_std_arr)
    species_data = run_parameter_swipe_parallel(n_cs_arr, K_std_arr,c_arr, args.iterations, args.leakage, K_mean=args.K_mean, n_jobs=-1)


    # Save final abundance matrix and parameters
    all_C = {f'C_{i}_{j}':species_data[i][j]['C'] for i in range(len(species_data)) for j in range(len(species_data[i]))}
    all_D = {f'D_{i}_{j}':species_data[i][j]['D'] for i in range(len(species_data)) for j in range(len(species_data[i]))}
    all_K = {f'K_{i}_{j}':species_data[i][j]['K'] for i in range(len(species_data)) for j in range(len(species_data[i]))}
    all_N_final = np.zeros((len(species_data), args.iterations, 4))*np.nan
    all_R_final = np.zeros((len(species_data), args.iterations, args.cs_max))*np.nan
    for i in range(len(species_data)):
        for j in range(len(species_data[i])):
            all_N_final[i, j, :] = species_data[i][j]['N_final']
            all_R_final[i, j, :len(species_data[i][j]['R_final'])] = species_data[i][j]['R_final']

    save_dict = {
        'all_N_final': all_N_final,
        'all_R_final': all_R_final,
        **all_C,
        **all_D,
        **all_K
    }

    # Save final abundance amtrix and parameters
    Path(args.folder).mkdir(exist_ok=True, parents = True)

    fn_1 = Path(args.folder) / f'{timestr}_data.npz'
    fn_2 = Path(args.folder) / f'{timestr}_args.txt'
    # fn_3 = Path(args.folder) / f'{timestr}_species_data.pkl'

    # Save as a dict


    with open(fn_1, 'wb') as f:
        np.savez(f, **save_dict)

    with open(fn_2, 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    # with open(fn_3, 'wb') as f:
    #     pickle.dump(species_data, f)

    total_seconds = time.time()-t0
    total_minutes = total_seconds/60
    print(f"Total minutes: {total_minutes:.2f}")
    print(n_cs_arr)
    print(K_std_arr)
    print(fn_1)
    print(fn_2)


