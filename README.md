![github_heading](github_heading.png)

# Robust coexistence in simple microbial communities

This repository contains data, code, and analysis notebooks for a study of robust coexistence in simple microbial communities. The work examines how four bacteria assemble and coexist across varying carbon source environments, and uses a consumer-resource model (CRM) to explore the mechanistic basis of coexistence.

## Species

| Abbreviation | Species |
|---|---|
| At | *Agrobacterium tumefaciens* |
| Ct | *Comamonas testosteroni* |
| Ml | *Microbacterium liquefaciens* |
| Oa | *Ochrobactrum anthropi* |

## Repository structure

```
data/                   Experimental data, organized by experiment
notebooks/              Jupyter notebooks for analysis and figure generation
python/                 Python scripts for consumer-resource model (CRM) simulations
simulation_results/     Output files from CRM parameter sweeps and dFBA simulations
```

## Reproducing the analysis

The numbered folders in `data/`, `notebooks/`, and `Figures/` follow the same ordering:

1. Growth phenotyping of monocultures
2. First community assembly experiment
3. Second community assembly experiment
4. Chemostat coexistence experiments
5. Pairwise interaction measurements
6. Consumer-resource model (CRM) parameterization and simulation

Run notebooks in order within each section. CRM parameter sweeps (section 6) are run via scripts in `python/6_CRM/` and results are saved to `simulation_results/`.

## Dependencies

- Python 3.x
- `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`
- `joblib` (for parallel CRM sweeps)
- Jupyter for notebooks

## Cite our work
Please cite this preprint: 
