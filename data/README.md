# data/

Experimental data organized by experiment. Subfolder numbering matches the corresponding notebook sections.

## 1_growth_phenotyping/

Monoculture growth characterization of all four species across a panel of carbon sources.

| File | Description |
|---|---|
| `od_all_species_long.csv` | Raw OD measurements in long format |
| `fitted_growth_parameters.csv` | Fitted growth rate and yield parameters per species and carbon source |
| `growth_no_growth.csv` | Binary growth/no-growth classification per condition |
| `growth_no_growth_p_value.csv` | Statistical significance of growth classifications |
| `yield.csv` | Yield estimates per species and carbon source |
| `selected_carbon_sources.csv` | Carbon sources selected for community experiments |
| `plate_mapping.csv` | Well-plate layout |
| `carveme_universe_bacteria.xml` | BIGG universal metabolic model used to choose carbon sources|
| `experimental_data/` | Raw plate reader files |

## 2_first_community_assembly/

CFU and plate reader data from the first community assembly experiment.

| File | Description |
|---|---|
| `cfus.csv` | Colony-forming unit counts per species and condition |
| `df_mean78.csv` / `df_logmean78.csv` | Mean and log-mean abundance summaries |
| `pca.csv` | PCA scores |
| `shannon_index.csv` | Shannon diversity index |
| `wellmap.csv` | Plate well layout |
| `platereader/` | Raw plate reader files |

## 3_second_community_assembly/

CFU and plate reader data from the second community assembly experiment.

| File | Description |
|---|---|
| `cfus.csv` | Colony-forming unit counts per species and condition |
| `wellmap.csv` | Plate well layout |
| `platereader/` | Raw plate reader files |

## 4_chemostats/

CFU and OD measurements from chemostat coexistence experiments.

| File | Description |
|---|---|
| `cfus.csv` | Colony-forming unit counts over time |
| `ods.csv` | Optical density measurements over time |

## 5_interactions/

Pairwise interaction measurements between species in fresh and spent media on malate and mannose / isoleucine.

| File | Description |
|---|---|
| `all_interactions_cfus.csv` | CFU data for all pairwise interactions |
| `all_interactions_cfus_with_stats.csv` | Same, with statistical test results |
| `cfus_fresh_malate.csv` | CFUs in fresh malate medium |
| `cfus_fresh_mannose.csv` | CFUs in fresh mannose medium |
| `cfus_spent_malate.csv` | CFUs in spent malate medium |
| `cfus_spent_mannose.csv` | CFUs in spent mannose medium |
| `platereader/` | Raw plate reader files |

## 6_CRM/

Empirical uptake rate data used to parameterize the consumer-resource model.

| File | Description |
|---|---|
| `sulheim_2025_rates.csv` | Per-species uptake rates for model parameterization |
