from pathlib import Path
from niceGAMEpy2 import niceGAME

model_files = {
    # '1w': '1w_GCA_040790105.xml',
    # '101': '101_GCA_041075215.xml',
    # '103': '103_GCA_041075265.xml',
    # '108': '108_GCA_040790455.xml',
    # '109': '109_GCA_041075275.xml',
    # '114': '114_GCA_040790125.xml',
    # '119': '119_GCA_041075285.xml',
    # '122': '122_GCA_040790085.xml',
    'At': 'At_GCA_030505215.xml',
    'Ct': 'Ct_GCA_030505195.xml',
    'Ml': 'Ml_GCA_030518755.xml',
    'Oa': 'Oa_GCA_030518775.xml',
}
auxotrophy_dict = {
'Ml': {'amino acids': ['cys__L'], 'vitamins':['thm', 'btn']},#'pro__L', 
'Oa': {'vitamins': ['thm']},
# '1w': {'vitamins': ['thm', 'btn'], 'amino acids': ['his__L', 'val__L', 'tyr__L', 'arg__L', 'met__L', 'thr__L', 'ile__L']}, # Also niacin, but not in the bigg universe
# '108': {'vitamins': ['thm', 'btn', 'pheme'], # Also niacin and phylloquinone (k1), neither in universe
#         'amino acids': ['his__L','adn', 'cytd', 'gsn', 'ins', 'thymd', 'ura']}, #adenosine + cytidine + guanosine + inosine + thymidine + uracil
}

print(Path.home())
repo_folder = Path("../..")
data_folder = repo_folder / "data" / "7_GEM_reconstruction"
carveme_draft_folder = data_folder / 'models/carveme'
growth_data_folder = repo_folder / 'data' / '1_growth_phenotyping'
binary_growth_data_path = growth_data_folder / 'growth_no_growth.csv'
carbon_source_ids_path = growth_data_folder / 'selected_carbon_sources.csv'#'carbon_source_ids_curated.csv'

M9_minimal_media_file = data_folder / 'M9_minimal_media_bigg.csv'
# vitamins_file = gapfilling_data_folder / 'vitamins_bigg.csv'
bigg_universe_fn = data_folder /'carveme_universe_bacteria_fixed.xml'#'bigg_universe.xml'
compartment_data_fn = data_folder /'compartment_data.json'
add_TFA = False # Include thermodynamics in gapfilling
gapfilled_model_folder  = data_folder / 'models' / 'gapfilled_FBA'


N = niceGAME(bigg_universe_fn, M9_minimal_media_file, binary_growth_data_path, carbon_source_ids_path, compartment_data_fn)

if add_TFA:
    N.get_universe_reactions_delta_G_from_equilibrator()
# cs_slack_dict, all_slacks = N.relax_universe()
N.set_model_folder(carveme_draft_folder)

N.set_auxotrophy_dict(auxotrophy_dict)
N.load_gapfill_solutions()
for species_abbr, model_name in model_files.items():
    N.load_model(species_abbr, model_name=model_name)
    if not N.gapfill_solutions.get(species_abbr):
        N.gapfill_model_on_all_cs(species_abbr, N_alternative_solutions = 10, add_TFA=False)
    N.store_gapfill_solutions()
    N.select_gapfill_solutions_and_gapfill(species_abbr)
    N.check_auxotrophies(species_abbr)
    gf_model_fn = gapfilled_model_folder / f'{species_abbr}_gapfilled.xml'
    N.save_gf_model(species_abbr, simulation_ready = True, fn = gf_model_fn)



