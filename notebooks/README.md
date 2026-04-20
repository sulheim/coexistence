# notebooks/

Jupyter notebooks for data analysis and figure generation. Subfolder numbering matches the corresponding `data/` and `Figures/` sections.

## 1_growth_phenotyping/

| Notebook | Description |
|---|---|
| `1A_plot_ml_oa_auxotrophy_screen.ipynb` | Auxotrophy screening for *Ml* and *Oa* |
| `1B_choose_carbon_sources.ipynb` | Selection of carbon sources for community experiments |
| `1C_CS_growth_phenotyping_4_MWF.ipynb` | Carbon source growth phenotyping for all four species |
| `1D_extract_growth_parameters_At_Ct_Ml_Oa.ipynb` | Fitting and extracting growth rate and yield parameters |
| `1E_plot_clustermaps.ipynb` | Clustermaps of growth rate and yield across species and carbon sources |

## 2_assembly_1/

| Notebook | Description |
|---|---|
| `2A_analyse_assembly_1.ipynb` | Statistical analysis of the first community assembly experiment |
| `2B_analyse_assembly_2.ipynb` | Follow-up analysis of assembly experiment 1 |
| `2C_plot_assembly_1_cfus.ipynb` | Plotting CFU data from assembly experiment 1 |

## 3_assembly_2/

| Notebook | Description |
|---|---|
| `3A_analyse_second_assembly.ipynb` | Analysis and plotting of the second community assembly experiment |
| `3B_plot_assembly_2_cfus.ipynb` | CFU plots for the second assembly experiment |

## 4_chemostats/

| Notebook | Description |
|---|---|
| `4A_plot_chemostat_assembly_cfus.ipynb` | CFU and OD plots for chemostat coexistence experiments |

## 5_interactions/

| Notebook | Description |
|---|---|
| `5A_plot_interactions.ipynb` | Plotting pairwise interaction outcomes |
| `5B_plot_interactions2.ipynb` | Additional pairwise interaction plots |

## 6_CRM/

Consumer-resource model parameterization, simulation, and analysis. These notebooks depend on the Python scripts in `python/6_CRM/` and the simulation outputs in `simulation_results/`.

| Notebook | Description |
|---|---|
| `6A_extract_parameters.ipynb` | Extract and fit CRM parameters from empirical data |
| `6B_plot_4_species_assembly.ipynb` | Simulate and plot 4-species assembly with the CRM |
| `6C_change_resource.ipynb` | Explore the effect of changing the primary resource in the CRM |
| `6D_plot_cs_screen.ipynb` | Plot results from the carbon source number parameter sweep |
| `6E_community_properties.ipynb` | Analyse emergent community properties from CRM simulations |
