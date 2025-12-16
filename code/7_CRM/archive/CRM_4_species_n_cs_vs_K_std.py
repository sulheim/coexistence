import numpy as np
import argparse
import json
import time
from pathlib import Path
from joblib import Parallel, delayed
import scipy.stats as st
from tqdm import tqdm

from CRM import CRM
from CRM_utils import make_D, numerical_error, has_converged



def run_simulation(n_cs, K_std, iterations, leakage, initial_c_conc, initial_abundance,
                   time, dilution_rate, n_species, K_mean, transfer=False, 
                   dt = 0.1,
                   method='BDF',
                   n_transfers=10, transfer_dilution=100, transfer_time=24):
    """
    Run a simulation with the specified parameters and return the final abundance matrix.
    If transfer is True, run the transfer simulation; otherwise, run the standard CRM simulation.
    """
    final_abundance_matrix = np.zeros((iterations, n_species))

    # Fixed arrays
    R0 = np.zeros(n_cs)
    R0[0] = initial_c_conc
    l = leakage * np.ones(n_species)
    N0 = np.ones(n_species) * initial_abundance    

    runtime_errors = 0
    k = 0
    k_control = 0
    k_numerical_error = 0
    k_not_converged = 0

    while k < iterations:
        C = np.zeros((n_species, n_cs))
        while C[:, 0].max() < 2 * dilution_rate:
            C = st.gamma.rvs(1.2, 0.01, 0.16, size=(n_species, n_cs))
            C = C / C.max()

        D = make_D(n_species, n_cs)
        K = np.random.lognormal(K_mean, K_std, (n_species, n_cs))
        c = CRM(n_species, n_cs, C, D=D, dilution_rate=dilution_rate, l=l, K=K)

        try:
            if transfer:
                sol = c.run_transfers(transfer_time, N0, R0, dt=dt,
                                      transfer_dilution=transfer_dilution,
                                      n_transfers=n_transfers, method=method)
            else:
                sol = c.run(time, N0, R0, dt=dt, method=method)
        except (RuntimeError, ValueError):
            runtime_errors += 1
        else:
            if transfer:
                final_abundance_matrix[k, :] = c.N[-1, :]
                k += 1
            else:
                if has_converged(c.N) and not numerical_error(c.N):
                    final_abundance_matrix[k, :] = c.N[-1, :]
                    k += 1
                else:
                    if numerical_error(c.N):
                        k_numerical_error += 1
                    if not has_converged(c.N):
                        k_not_converged += 1
        k_control += 1
        if k_control > iterations * 3:
            print("Too many iterations without convergence, breaking loop")
            print(f"Runtime errors encountered: {runtime_errors}")
            print(f"Numerical errors encountered: {k_numerical_error}")
            print(f"Not converged: {k_not_converged}")
            print(f"Total iterations attempted: {k_control}")
            print(f"Successful iterations: {k}")
            break

    return final_abundance_matrix

def run_parameter_swipe_parallel(n_cs_arr, K_std_arr, iterations, leakage, initial_c_conc=10, 
                                  initial_abundance=1e-2, time=2000, dilution_rate=0.1, 
                                  n_species=4, K_mean=-1, transfer = False, n_jobs=-1):
    
    params_list = []
    for n_cs in n_cs_arr:
        for K_std in K_std_arr:
            params_list.append([n_cs, K_std, iterations, leakage, initial_c_conc, initial_abundance,
                                time, dilution_rate, n_species, K_mean, transfer])
    
    num_jobs = len(params_list)
    if transfer:
        print('Run transfer CRM simulation')
    else:
        print('Run chemostat CRM simulation')
    results = Parallel(n_jobs=n_jobs)(
        tqdm(
            (delayed(run_simulation)(*params) for params in params_list),
            total=len(params_list)
        )
    )
    
    print(f"Finished {num_jobs} jobs")
    # Reshape results into final abundance matrix
    final_abundance_matrix = np.array(results).reshape(len(n_cs_arr), len(K_std_arr), iterations, n_species)
    
    return final_abundance_matrix


if __name__ == '__main__':
    t0 = time.time()
    parser = argparse.ArgumentParser(description = 'Parameter swipe')
    parser.add_argument("--cs_min", help = 'Minimal number of CS', default=1, type=int)
    parser.add_argument("--cs_max", help = 'Maximal number of CS', default=50, type=int)
    parser.add_argument("--N_cs", help = 'The number of different sizes of CS', default=6, type=int)
    parser.add_argument("--K_mean", help = 'Mean K', default = -1, type=float)
    parser.add_argument("--K_std_min", help = 'Minimal K std', default = 0, type=float)
    parser.add_argument("--K_std_max", help = 'Maximal K std', default = 4, type=float)
    parser.add_argument("--N_K_std", help = 'Number of different K std values', default = 6, type=int)
    parser.add_argument("--iterations", help = 'Number of iterations per parameter combination', default = 10, type=int)
    parser.add_argument("--leakage", help = "Leakage fraction", default=0.1, type=float)
    parser.add_argument("--folder", help = 'Where to store data', default = './n_cs_vs_K_parameter_swipe', type=str)
    parser.add_argument("--transfer", help = 'Run serial transfers', default = False, type=bool)
    
    args = parser.parse_args()
    timestr = time.strftime("%Y%m%d-%H%M%S")

    n_cs_arr = np.linspace(args.cs_min, args.cs_max, num=args.N_cs, dtype=int, endpoint = True)
    K_std_arr = np.linspace(args.K_std_min, args.K_std_max, num = args.N_K_std, endpoint = True)
    print(n_cs_arr)
    print(K_std_arr)
    final_abundance_matrix = run_parameter_swipe_parallel(n_cs_arr, K_std_arr, args.iterations, args.leakage, K_mean=args.K_mean, transfer=args.transfer)

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
    print(K_std_arr)


