# python/

Python scripts for running Consumer-Resource Model (CRM) simulations.

## 6_CRM/

| File | Description |
|---|---|
| `CRM.py` | Core CRM implementation. Defines the `CRM` class, which integrates the consumer-resource ODEs (with leakage and cross-feeding) using `scipy.solve_ivp`. |
| `CRM_utils.py` | Utility functions: convergence checks, richness calculation, and helpers for generating random preference matrices (`make_C`, `make_D`). |
| `CRM_4_species_cs_sweep.py` | Parameter sweep over the number of carbon sources. Runs parallel simulations with `joblib` and saves results to `simulation_results/cs_sweep/`. |
| `CRM_4_species_K_Ncs_C_sweep.py` | Parameter sweep over half-saturation constant (K), number of carbon sources, and preference matrix sparsity. Saves results to `simulation_results/n_cs_vs_K_parameter_swipe/`. |

## Running simulations

Scripts are run from the `python/6_CRM/` directory and accept command-line arguments (parsed with `argparse`). Output is written to `simulation_results/` with a timestamp prefix (`YYYYMMDD-HHMMSS`). Each run produces an `_args.txt` file recording the parameters used and a `_data.npz` file with simulation results.

```bash
cd python/6_CRM
python CRM_4_species_cs_sweep.py --help
```
