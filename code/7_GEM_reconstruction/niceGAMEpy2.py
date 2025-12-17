# -*- coding: utf-8 -*-
"""
module: niceGAMEpy2

description:

author: Snorre Sulheim
date: October 16, 2023

"""
import pandas as pd
from pathlib import Path
import numpy as np
import time
import json
import logging
import sys
from collections import defaultdict
import time
import random
import dotenv

import reframed
reframed.set_default_solver('gurobi')
from reframed.solvers import solver_instance
from reframed.solvers.solution import Status
from reframed.core.transformation import disconnected_metabolites
from reframed.solvers.solver import VarType, Parameter


from gurobipy import Model as GurobiModel, GRB
# import xmltodict
from equilibrator_api import ComponentContribution, Q_
from ng_utils import *
from ng_tests import test_model
from fix_universe import fix_misc, add_biotin_synthase_reactions
from model_qc_and_polish import curate_model, fix_biomass, polish_model


##### ADD REACTIONS FROM iJO1366 to allow biotin production

REPO_PATH =  Path(dotenv.find_dotenv()).parent
TMP_FOLDER = REPO_PATH / 'tmp'
TMP_FOLDER.mkdir(exist_ok=True)
timestr = time.strftime("%Y%m%d_%H%M")
LOGFILE = TMP_FOLDER / f'niceGAMEpy_{timestr}.log'
# logging.basicConfig(filename='nicegame.log', encoding='utf-8', level=logging.DEBUG
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(levelname)s:%(name)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                     handlers=[logging.FileHandler(str(LOGFILE)),logging.StreamHandler()])
logging.captureWarnings(True)

# VITAMINS_FN = REPO_PATH / 'gapfilling_data' / 'vitamins_bigg.csv'
# VITAMINS = ['thm', 'btn', 'ribflv', 'pydx', 'cbl1', 'lipoate', 'fol', 'pnto__R', 'nac', '4abz']
DEFAULT_CS = ['glc__D', 'cit', 'ac', 'fru', 'pyr']


class gapfillResults(object):
    def __init__(self, species_abbr):
        self.species_abbr = species_abbr
        self.gapfill_solutions = None
        self.consensus_reactions = None
        self.specific_gapfill_reactions = None
        self.added_reactions = None
        self.removed_reactions = None
        self.confusion_matrix = None
        self.df = None

    def score(self):
        if self.confusion_matrix:
            return MCC(self.confusion_matrix)
        else:
            return None

    def save(self):
        fn = TMP_FOLDER / f'gapfill_results_{self.species_abbr}.json'
        with open(fn, 'w') as f:
            save_dict = {key: value for key, value in self.__dict__.items() if is_jsonable(value)}
            save_dict['df_dict'] = self.df.to_json(orient = 'split')
            json.dump(save_dict, f, sort_keys=True)

    def load(self):
        fn = TMP_FOLDER / f'gapfill_results_{self.species_abbr}.json'
        with open(fn, 'r') as f:
            # save_dict = {key: value for key, value in self.__dict__.items() if not key =='df'}
            # save_dict['df_dict'] = self.df.to_json(orient = 'split')
            dic = json.load(f)
            self.__dict__.update(dic)

class niceGAME(object):
    def __init__(self, universe_fn, base_medium_fn, binary_growth_fn, 
                 carbon_source_id_mapping_fn, compartment_data_fn, vitamins_fn = None):

        self.bigM = 1e3
        self.abstol = 1e-9
        self.intfeastol = 1e-9
        self.solver_time_limit = 1 #minutes
        self.solver_name = 'gurobi'
        self.db = 'BiGG'

        self.min_concentration = 1e-6
        self.max_concentration = 1e-1
        self.default_cs_lb = -10
        self.default_min_growth = 0.1

        # Vitamins can sometimes be used as carbon sources (in the universe), 
        # and it can therefore mess up with the gap-fill-solutions
        # Unless you know what you are doing - don't change this
        self.gapfill_with_vitamins = False

        # Default TFA parameters
        self.ignore_model_bounds = False
        self.gapfill_solutions = {}
        self.gapfill_results = {}
        self.gapfilled_models = {}
        self.cobra_method = 'TFA'


        self.vitamins_fn = None
        self.base_medium_fn = None
        self.test_universe_growth_reaction = 'R_EX_glc__D_e'
        self.model_solvers = {}
        self.models = {}
        self.auxotrophy_dict = {}
        self.biotin_auxotrophs = []
        self.auxotrophy_aa_uptake = -0.2
        self.auxotrophy_vitamin_uptake = -1e-3


        self.positive_growth_mets = {}
        self.positive_growth_met_ids = {}


        self.init_base_medium(vitamins_fn, base_medium_fn)
        self.init_growth_data(binary_growth_fn, carbon_source_id_mapping_fn)
        self.load_and_prep_universe(universe_fn)
        self.load_compartment_data(compartment_data_fn)



    def init_solver(self, solver):
        # reframed.set_default_solver(self.solver_name)
        solver.problem.params.IntFeasTol = self.intfeastol
        solver.problem.params.IntegralityFocus = 1
        solver.problem.params.MIPFocus = 2
        solver.problem.params.FeasibilityTol = self.abstol
        solver.problem.params.OptimalityTol = self.abstol
        solver.problem.params.TimeLimit = self.solver_time_limit * 60 # Convert to seconds

    def init_base_medium(self, vitamins_fn = None, base_medium_fn = None):
        """
        Read in the non-carbon components of the medium
        """
        self.vitamins_fn = vitamins_fn
        self.base_medium_fn = base_medium_fn

        self.base_metabolites = list(pd.read_csv(base_medium_fn, header = None)[0])
        
        if vitamins_fn:
            self.vitamins = list(pd.read_csv(vitamins_fn, header = None)[0])

        if vitamins_fn and self.gapfill_with_vitamins:
            self.base_medium = self.vitamins+self.base_metabolites
        else:
            self.base_medium = self.base_metabolites

    def init_growth_data(self, binary_growth_fn, carbon_source_id_mapping_fn):
        self.binary_growth_data = pd.read_csv(binary_growth_fn, index_col=0)
        self.cs_name_to_id = pd.read_csv(carbon_source_id_mapping_fn, index_col=1).to_dict()[f'{self.db} ID']
        self.any_growth_carbon_sources = list(self.binary_growth_data.index[self.binary_growth_data.any(axis =1)])
        self.carbon_sources = [x for x in self.binary_growth_data.index]
        self.carbon_source_ids = [self.cs_name_to_id[x] for x in self.carbon_sources]



    def load_and_prep_universe(self, universe_fn):
        """
        Adds biotin to the biomass function
        """
        self.universe_fn = universe_fn
        logger = logging.getLogger('universe')
        logger.info('Loading universe')
        self.universe =  reframed.load_cbmodel(universe_fn)

        # Init universe solver
        solver = solver_instance(self.universe)
        self.init_solver(solver)
        self.model_solvers['universe'] = solver


        # Prep universe
        # fix_reframed_annotations(self.universe)
        fix_compartments(self.universe)
        fix_biomass(self.universe)
        add_biotin_synthase_reactions(self.universe, logger = logger)
        fix_misc(self.universe)
        curate_model(self.universe, logger = logger)


        if self.vitamins_fn and self.base_medium_fn:
            # Add missing exchanges and set medium
            add_missing_exchanges(self.universe, self.base_medium+self.carbon_source_ids, metabolite_id_prefix = 'M_')#'vitamins_bigg.csv'
            set_base_environment(self.universe, self.base_metabolites)

        # Verify growth on carbon_source
        sol = reframed.FBA(self.universe, constraints={self.test_universe_growth_reaction: self.default_cs_lb})
        logger.debug(f"Universe growth on glucose: {sol}")
        
        # Get some useful universe parameters
        universe_size = get_model_size(self.universe)
        logger.info('Succesfully loaded universe with this content: %s', universe_size)
        

    def load_compartment_data(self, compartment_data_fn):
        with open(compartment_data_fn, 'r') as f:
            compartment_data = json.load(f)
        # compartment_data = read_compartment_data(str(pytfa_folder / 'models/iJO1366/compartment_data.json'
        # Membrane potential from e to c similar to p to c
        compartment_data['c']['membranePot']['e'] = compartment_data['c']['membranePot']['p']
        compartment_data['e']['membranePot']['c'] = compartment_data['p']['membranePot']['c']
        self.compartment_data = compartment_data

    def set_auxotrophy_dict(self, auxotrophy_dict):
        logger = logging.getLogger('auxotrophies')
        constraints_dict = {}
        for species_abbr, species_dict in auxotrophy_dict.items():
            species_constraint_dict = {}
            if species_dict.get('vitamins'):
                for met_id in species_dict['vitamins']:
                    r_id = f'R_EX_{met_id}_e'
                    species_constraint_dict[r_id] = (self.auxotrophy_vitamin_uptake, 0)
                    if met_id == 'btn':
                        self.biotin_auxotrophs.append(species_abbr)
            if species_dict.get('amino acids'):
                for met_id in species_dict['amino acids']:
                    r_id = f'R_EX_{met_id}_e'
                    species_constraint_dict[r_id] = (self.auxotrophy_aa_uptake, 0)
            constraints_dict[species_abbr] = species_constraint_dict
        logger.info('Defined auxotrophy constraints: %s', constraints_dict)        
            
        self.auxotrophy_constraints = constraints_dict
        self.auxotrophy_dict = auxotrophy_dict


    def set_model_folder(self, path):
        self.model_folder = path

    def load_model(self, species_abbr, model_name = None):
        if not model_name:
            model_name = f'{species_abbr}.xml'
        model_fn = self.model_folder / model_name
        logger = logging.getLogger(f'loading.{model_fn}')
        model = reframed.load_cbmodel(model_fn)

        logger = logging.getLogger(f'loaded.{species_abbr}')
        
        model.solver = 'gurobi'
        logger = logging.getLogger('Set solver to gurobi')
        # Init umodel solver
        solver = solver_instance(model)
        self.init_solver(solver)

        # Prep model
        fix_compartments(model)
        # fix_reframed_annotations(model)
        curate_model(model, logger = logger)
        fix_biomass(model, universe = self.universe, add_biotin = True)
        if not species_abbr in self.biotin_auxotrophs:
            add_biotin_synthase_reactions(model, logger=logger)

        positive_growth_mets = list(self.binary_growth_data.index[self.binary_growth_data[species_abbr]==1])
        positive_growth_met_ids = [self.cs_name_to_id[x] for x in positive_growth_mets]
        # cs_with_e = [f'{x}_e' for x in positive_growth_met_ids]

        add_missing_exchanges(model, self.base_medium+self.carbon_source_ids, metabolite_id_prefix = 'M_', universe = self.universe)
        set_base_environment(model, self.base_metabolites)

        # Check that the model has all the exchanges self.auxotrophy_constraints[species_abbr]:
        if self.auxotrophy_constraints.get(species_abbr):
            add_missing_exchanges_from_reactions_ids(model, list(self.auxotrophy_constraints[species_abbr].keys()), universe = self.universe)

        # Set ATP maineinance equal to e. coli value in iJO1366
        model.reactions['R_ATPM'].lb = 3.15 
        logger.info(f'Model size: {len(model.reactions)} reactions, {len(model.metabolites)} metabolites')

        self.positive_growth_mets[species_abbr] = positive_growth_mets
        self.positive_growth_met_ids[species_abbr] = positive_growth_met_ids
        self.model_solvers[species_abbr] = solver
        self.models[species_abbr] = model

        return model

    def gapfill_model_on_all_cs(self, species_abbr, N_alternative_solutions = 1, add_TFA = True, min_growth = 0.1):
        """
        Todo: Perform gapfilling based on estimated growth yields

        """
        logger = logging.getLogger(f'gapfill.{species_abbr}')
        model = self.models[species_abbr].copy()
        if add_TFA:
            logger.info(f'Starting gapfilling for {species_abbr} with TFA')
        else:
            logger.info(f'Starting gapfilling for {species_abbr} with FBA')

        # Merge and prep model
        gf_model, new_reactions = _prep_for_gapfill(model, self.universe, self.ignore_model_bounds, self.deltaG0)


        # Get list of carbon sources where the organism should grow
        carbon_sources = self.positive_growth_mets[species_abbr]

        # Get solver
        solver = solver_instance(gf_model)
        self.init_solver(solver)

        
        constraints = {self.test_universe_growth_reaction: self.default_cs_lb}
        
        if self.auxotrophy_dict.get(species_abbr):
            logger.info(f'{species_abbr} is an auxotroph for: %s', self.auxotrophy_dict[species_abbr])
            auxotrophy_constraints = self.auxotrophy_constraints[species_abbr]
        else:
            auxotrophy_constraints = {}

        # Add TFA constraints
        if add_TFA:
            self.cobra_method = 'TFA'
            logger.info('Adding TFA variables and constraints')
            lhs_dict = _add_TFA_constraints(gf_model, solver, self.deltaG0, self.sdeltaG0, 
                            concentration_min=self.min_concentration,
                            concentration_max=self.max_concentration, bigM = self.bigM)

            nv, nc = len(solver.problem.getVars()), len(solver.problem.getConstrs())
            logger.info(f"Number of variables and constraints after TFA-preparation: {nv}, {nc}")
            # if is_auxotroph:
            #     constraints.update({key: value for key, value in auxotrophy_constraints.items() if not constraints.get(key)})
            # logger.info(solver.solve(linear={'Growth':1}, minimize=False, get_values=False, constraints=constraints))
        # Add gapfill variables and constraints
        _add_gapfilling_constraints(gf_model, solver, new_reactions, min_growth, self.bigM, use_indicator_constraints = False)
        nv, nc = len(solver.problem.getVars()), len(solver.problem.getConstrs())
        logger.info(f"Number of variables and constraints after gapfill-preparation: {nv}, {nc}")
        # Set objective
        gf_objective = {'z_'+r_id: 1.0 for r_id in new_reactions}
    
        # Generate multiple solutions
        added_cut_constraints = []
        cs_alt_solutions = {}
        for i, cs in enumerate(carbon_sources):
            alt_gf_solutions = []
            cs_id = self.cs_name_to_id[cs]
            cs_exchange_id = f'R_EX_{cs_id}_e'
            constraints = {cs_exchange_id: self.default_cs_lb}

            alt_gf_solutions = []
            if len(auxotrophy_constraints):
                constraints.update({key: value for key, value in auxotrophy_constraints.items() if not constraints.get(key)})
            
            if len(added_cut_constraints):
                logger.debug(f'Clear cut constraints %s', added_cut_constraints)
                # Clear cut constraints
                nv, nc = len(solver.problem.getVars()), len(solver.problem.getConstrs())
                logger.debug(f"Number of variables and constraints before removing cut constraints: {nv}, {nc}")
                for constr in added_cut_constraints:
                    print(constr)
                    # solver.problem.remove(constr)
                    solver.remove_constraint(constr)
                solver.update()
                nv, nc = len(solver.problem.getVars()), len(solver.problem.getConstrs())
                logger.debug(f"Number of variables and constraints after removing cut constraints: {nv}, {nc}")
                added_cut_constraints = []
                
            for j in range(N_alternative_solutions):
                logger.info(f'Creating solution {j+1} for {cs}')
                if j!=0:
                    # Add cut constraints
                    added_rxns_use_vars = ['z_'+r_id for r_id in added_reactions]
                    # sum_use_var = np.sum([solution.values[x] for x in added_rxns_use_vars])
                    solver.add_constraint(f'cut_{str(i)}_{str(j)}', {x:1 for x in added_rxns_use_vars},
                                          '<', len(added_reactions)-1) # < is less or equal
                    
                    added_cut_constraints.append(solver.problem.getConstrByName(f'cut_{str(i)}_{str(j)}'))
                    # added_cut_constraints.append('cut_'+str(j))
                solver.update()
                
                solution = solver.solve(objective=gf_objective, minimize=True, constraints=constraints)
                logger.debug(f'Gapfill-solution: %s', solution)

                # Find added reactions
                inactive, added_reactions, failed = _get_added_gf_reactions(solution, new_reactions, self.abstol, tag = cs, logger = logger)
                if failed:
                    break
                else:            
                    alt_gf_solutions.append(added_reactions)
                    logger.debug('Added reactions: ', ', '.join(added_reactions))
   
            cs_alt_solutions[cs] = alt_gf_solutions

        self.gapfill_solutions[species_abbr] = cs_alt_solutions

        return gf_model, cs_alt_solutions

    def store_gapfill_solutions(self, filename = None):
        if not filename:
            filename = TMP_FOLDER / 'gapfill_solutions.json'
        logger = logging.getLogger('gapfill')
        if len(self.gapfill_solutions):
            with open(filename, 'w') as f:
                json.dump(self.gapfill_solutions, f)
            logger.info('Stored gapfill solutions at %s', str(filename))
            return True
        else:
            logger.info('Did not store gapfill solutions because there was no data')
            return False

    def load_gapfill_solutions(self, filename = None, overwrite = False):
        if not filename:
            filename = TMP_FOLDER / 'gapfill_solutions.json'
        logger = logging.getLogger('gapfill')
        loaded = False
        if (not len(self.gapfill_solutions)) or overwrite:
            try:
                with open(filename, 'r') as f:
                    self.gapfill_solutions = json.load(f)
            except FileNotFoundError:
                logger.info('Could not load gapfill solutions from %s', str(filename))
            else:
                logger.info('Loaded gapfill solutions from %s', str(filename))
                logger.info('This has solutions for: %s', ', '.join(self.gapfill_solutions.keys()))
                loaded = True
        return loaded

    def select_gapfill_solutions_and_gapfill(self, species_abbr, consensus_threshold = 0.5, test_method = None):
        """
        From all possible gap-filling solutions, this method selects a "minimum" according to the following procedure
        1. Add "consensus reactions", these are reactions that are suggested for more than a ratio (consensus threshold) of the carbons sources 
        2. Then remove these reactions from the suggested solutions
        3. Then add "specific" reactions, iteratively select the shortest solutions that solves the most carbon sources
        """
        if not test_method:
            test_method = self.cobra_method

        logger = logging.getLogger(f'gapfill.{species_abbr}')
        if not self.gapfill_solutions.get(species_abbr):
            logger.warning(f'No gapfill solutions for {species_abbr}')
            return False

        
        if self.auxotrophy_constraints.get(species_abbr):
            auxotrophy_constraints = self.auxotrophy_constraints[species_abbr]
        else:
            auxotrophy_constraints = {}

        gfR = gapfillResults(species_abbr)
        gfR.gapfill_solutions = self.gapfill_solutions[species_abbr]

        logger.info('Starting gapfilling')
        gf_model = self.models[species_abbr].copy()
        reduced_solution_dict, added_reactions = select_consensus_reactions(self.gapfill_solutions[species_abbr])
        logger.info('Reactions added after finding consensus reactions %s', added_reactions)
        gfR.consensus_reactions = added_reactions
        
        added_reactions, solved_cs = select_specific_gapfill_solutions(self.gapfill_solutions[species_abbr], reduced_solution_dict, added_reactions)
        logger.info('Reactions added after finding specific reactions %s', added_reactions)
        gfR.specific_gapfill_reactions = [x for x in added_reactions if not x in gfR.consensus_reactions]

        logger.info(f'Gapfilling complete, added {len(added_reactions)} reactions')
        logger.info(f'This solved {len(solved_cs)} of {len(self.gapfill_solutions[species_abbr])} carbon sources')

        logger.info('Test gapfilled model....')
        add_reactions_from_universe(gf_model, self.universe, added_reactions)
        test_df, confusion_matrix = test_model(gf_model, self.binary_growth_data[species_abbr], self.cs_name_to_id, additional_constraints=auxotrophy_constraints,
                                               method = test_method, deltaG0=self.deltaG0, sdeltaG0 = self.sdeltaG0)
        logger.info(test_df)
        nFP = confusion_matrix['FP']
        if nFP:

            logger.info(f'Start gap-maker, {nFP} false positives')
            positive_growth_met_ids = test_df.loc[test_df['In vitro'] == 1, 'Carbon source ID'].values
            reactions_to_remove = self.gap_maker(species_abbr, gf_model, test_df, confusion_matrix, positive_growth_met_ids, auxotrophy_constraints)
            logger.info(f'Removed {len(reactions_to_remove)} reactions')
            gf_model.remove_reactions(reactions_to_remove)
            gfR.removed_reactions = reactions_to_remove
            test_df, confusion_matrix = test_model(gf_model, self.binary_growth_data[species_abbr], self.cs_name_to_id, additional_constraints=auxotrophy_constraints,
                                                   method = test_method, deltaG0=self.deltaG0, sdeltaG0 = self.sdeltaG0)

        gfR.confusion_matrix = confusion_matrix
        gfR.df = test_df
        
        gfR.save()
        self.gapfill_results[species_abbr] = gfR
        self.gapfilled_models[species_abbr] = gf_model

    def check_auxotrophies(self, species_abbr, fix_auxotrophies = True):
        logger = logging.getLogger(f'auxotrophy.{species_abbr}')
        if not self.auxotrophy_constraints.get(species_abbr):
            logger.info(f'No auxotrophy information available for {species_abbr}')
            return True

        # Check if auxotrophies are correctly predicted
        
        auxotrophy_constraints = self.auxotrophy_constraints[species_abbr]
        r_id_to_met_id = {x: x.lstrip('R_EX_').rstrip('_e') for x in auxotrophy_constraints.keys()}
        
        gf_model = self.gapfilled_models[species_abbr]

        # Try to get precomputed information
        if self.gapfill_results.get(species_abbr):
            gfR =  self.gapfill_results[species_abbr]
            test_df = gfR.df
        else:
            test_df, _ = test_model(gf_model, self.binary_growth_data[species_abbr], self.cs_name_to_id, additional_constraints=auxotrophy_constraints, method = 'FBA')

        # Use the best-growing true-positive CS to test auxotrophies
        sorted_df = test_df.loc[test_df['Pred outcome'] == 'TP', :].sort_values(by='In silico', ascending = False)
        
        # Make sure that auxotrophy is not tested with the auxotrophy metabolites
        k = 0
        test_met = sorted_df.iloc[k]['Carbon source ID']
        auxotrophy_mets = list(r_id_to_met_id.values())
        while test_met in auxotrophy_mets:
            k +=1
            test_met = sorted_df.iloc[k]['Carbon source ID']
        logger.info(f'Testing auxotrophies with {test_met} as carbon source')


        # Test growth without the auxotrophy constraints
        zero_constraints = {f'R_EX_{test_met}_e': self.default_cs_lb}
        zero_solution = reframed.FBA(gf_model, constraints = zero_constraints)
        if zero_solution.fobj > self.default_min_growth:
            logger.info(f'Gapfilled {species_abbr} model grows without any of its auxotrophies in the medium. Not good.')
        else:
            logger.info(f'Gapfilled {species_abbr} does not grow without any of its auxotrophies in the medium. Good.')

        # Test one by one auxotrophy met
        except_one_met_results = {}
        incorrect_auxo = []
        for r_id, _ in auxotrophy_constraints.items():
            m_id = r_id_to_met_id[r_id]
            constraints = {key: value for key, value in auxotrophy_constraints.items() if key != r_id}
            constraints[r_id] = 0
            constraints.update(zero_constraints)

            solution = reframed.FBA(gf_model, constraints = constraints)
            except_one_met_results[r_id] = solution.fobj
            if solution.fobj > self.default_min_growth:
                logger.info(f'Gapfilled {species_abbr} model grows without {m_id} in the medium. Not good.')
                incorrect_auxo.append(r_id)
        # Try to
        logger.info(f'{len(incorrect_auxo)} incorrect auxotrophies: %s', incorrect_auxo)
        
        if fix_auxotrophies and len(incorrect_auxo):
            logger.info('Trying to fix auxotrophies by finding uniquely essential reactions')

            # Get all essential reactions
            cs_ids = list(test_df.loc[test_df['Pred outcome'] == 'TP', 'Carbon source ID'].values)
            essential_reactions, essential_reactions_dict = get_all_essential_reactions(species_abbr, gf_model, cs_ids, self.default_cs_lb, auxotrophy_constraints, self.default_min_growth, 
                                                                                     method = self.cobra_method, deltaG0=self.deltaG0, sdeltaG0 = self.sdeltaG0, logger = logger)
            excluded_reactions = list(set(essential_reactions + gf_model.get_exchange_reactions()))
            solutions = {}
            for r_id in incorrect_auxo:
                # Get essential reactions when this met is not available
                m_id = r_id_to_met_id[r_id]
                constraints = {key: value for key, value in auxotrophy_constraints.items() if key != r_id}
                constraints[r_id] = 0
                constraints.update(zero_constraints)
                except_one_essentials = get_essential_reactions(species_abbr, gf_model, self.default_min_growth, constraints, method = self.cobra_method, deltaG0=self.deltaG0, sdeltaG0 = self.sdeltaG0, logger = logger)

                # Potential targets based on list of essential reactions
                potential_targets = [x for x in except_one_essentials if not x in excluded_reactions]
                if len(potential_targets):
                    logger.info(f'{len(potential_targets)} identified to solve auxotrophy issue for {m_id}: %s', potential_targets)
                    scores_dict = rank_auxotrophy_solutions(gf_model, potential_targets, m_id)
                    if len(scores_dict):
                        solutions[r_id] = max(scores_dict, key=scores_dict.get)
                    else:
                        solutions[r_id] = None

            still_incorrect_auxo = [x for x in incorrect_auxo if not x in list(solutions.keys())]
            
            logger.info('Trying to fix auxotrophies by removing reactions producing the target compound')
            # Try to solve by removing reactions producing the compound
            for r_id in still_incorrect_auxo:
                m_id = r_id_to_met_id[r_id]
                potential_targets = [r_id for r_id in gf_model.get_metabolite_reactions(f'M_{m_id}_c') if not r_id in excluded_reactions]
                logger.info(f'{len(potential_targets)} identified to solve auxotrophy issue for {m_id}: %s', potential_targets)

                constraints = {key: value for key, value in auxotrophy_constraints.items() if key != r_id}
                constraints[r_id] = 0
                constraints.update(zero_constraints)

                removed_targets = []
                while True:
                    if not len(potential_targets):
                        break

                    curr_constraints = constraints.copy()
                    for t in potential_targets:
                        curr_constraints[t] = 0


                    curr_solution = reframed.FBA(gf_model, constraints = curr_constraints)
                    print(potential_targets, curr_solution.fobj)

                    if curr_solution.fobj < self.default_min_growth:
                        random.shuffle(potential_targets)
                        removed_targets.append(potential_targets.pop(0))
                        counter = 0
                    else:
                        if len(removed_targets):
                            counter += 1
                            potential_targets.append(removed_targets.pop(-1))
                            if counter > len(potential_targets):
                                break
                            else:
                                removed_targets.append(potential_targets.pop(0))
                        else:
                            break
                solutions[r_id] = potential_targets           
                
            for r_id in incorrect_auxo:
                m_id = r_id_to_met_id[r_id]
                if solutions.get(r_id):
                    logger.info(f'Removing these reactions to solve false-positive growth without {m_id}: %s', solutions[r_id])
                    if not isinstance(solutions[r_id], list):
                        solutions[r_id] = [solutions[r_id]]
                    gf_model.remove_reactions(solutions[r_id])
                else:
                    logger.info(f'No solution found to false-positive growth without {m_id}')


    

    def save_gf_model(self, species_abbr, model = None, fn = None, simulation_ready = True):
        logger = logging.getLogger(f'save.{species_abbr}')
        if not model:
            model = self.gapfilled_models[species_abbr]

        polish_model(model, logger)
        logger.info('Saving model ....')
        if simulation_ready:
            # Add carbon source and auxotrophy constraints
            if self.auxotrophy_constraints.get(species_abbr):
                for r_id, bounds in self.auxotrophy_constraints[species_abbr].items():
                    r = model.reactions[r_id]
                    model.set_flux_bounds(r_id, bounds[0], bounds[1])
                    logger.info(f'Set bounds of {r_id} to %s', bounds)
            # Add carbon source
            cs_idx = 0
            while cs_idx < len(DEFAULT_CS):
                m_id = DEFAULT_CS[cs_idx]
                r_id = f'R_EX_{m_id}_e'
                r = model.reactions[r_id]
                
                solution = reframed.FBA(model, constraints = {r_id: (self.default_cs_lb, 0)})
                if (solution.status == Status.OPTIMAL) and (solution.fobj > self.default_min_growth):
                    model.set_flux_bounds(r_id, self.default_cs_lb, 0)
                    logger.info(f'Set bounds of {r_id} to %s', (self.default_cs_lb, 0))
                    break
                else:
                    cs_idx += 1

        # SAve model
        if not fn:
            fn = TMP_FOLDER / f'gapfilled_{species_abbr}.xml'
        reframed.save_cbmodel(model, str(fn))



    def get_universe_reactions_delta_G_from_equilibrator(self, save = True, try_load = True):
        deltaG0_fn = TMP_FOLDER / 'eq_deltaG0.json'
        sdeltaG0_fn = TMP_FOLDER / 'eq_sdeltaG0.json'
        loaded_deltaG0 = False
        logger = logging.getLogger('equilibrator')
        if try_load:
            try:
                with open (deltaG0_fn, 'r') as f:
                    self.deltaG0 = json.load(f)
                with open (sdeltaG0_fn, 'r') as f:
                    self.sdeltaG0 = json.load(f)
            except FileNotFoundError:
                logger.info(f'Json files with thermodynamic data not found at {deltaG0_fn} and {sdeltaG0_fn}')    
            else:
                loaded_deltaG0 = True

        if not loaded_deltaG0:
            logger.info('Getting deltaG0 values from equilibrator')
            self.deltaG0, self.sdeltaG0 = get_dG_for_model_reactions_from_eq(self.universe, self.compartment_data)
        
        ndG = len(self.deltaG0)
        universe_size = get_model_size(self.universe)
        nr = universe_size['other']+universe_size['transport']
        logger.info(f'Obtained deltaG0 values for {ndG} reactions, that is {100*ndG/nr:.0f}% of the {nr} non-exchange reactions')


        # Truncate deltaG values
        self.deltaG0 = truncate_deltaG0_values(self.deltaG0, logger=logger)

        if save:
            for fn, file in zip([deltaG0_fn, sdeltaG0_fn], [self.deltaG0, self.sdeltaG0]):
                with open (fn, 'w') as f:
                    json.dump(file, f)
    def gap_maker(self, species_abbr, gf_model, test_df, confusion_matrix, positive_growth_met_ids, 
                auxotrophy_constraints = None, min_growth = 0.1, 
                uptake_rate = -10, exclude_exchanges = True):
        """
        # This is the procedure for generating gaps
        1. Find all reactions that are essential in any of the TP conditions
        2. Find the reactions that are uniquely essential in each FP conditions (with respect to the essential reactions in the TP cases)
        3. If any reaction shows up in multiple sets from 2. -> Delete that one
        4. if not, check if a transport reaction transporting the actual metabolite is the list from 2. Remove this one
        5. Else, remove any transport reaction in set from 2
        6. Else, remove any reaction from set 2
        7. If set from 2 is empty -> skip and let manual curation do this if necessary
        """

        
        logger = logging.getLogger(f'gapmaker.{species_abbr}')
        exchange_reactions = gf_model.get_exchange_reactions()
        logger.info('Finding essential reactions... might take a while')
        cs_ids = list(test_df.loc[test_df['Pred outcome'] == 'TP', 'Carbon source ID'].values)
        essential_reactions, essential_reactions_dict = get_all_essential_reactions(species_abbr, gf_model, cs_ids, uptake_rate, auxotrophy_constraints, min_growth, 
                                                                                method = self.cobra_method, deltaG0=self.deltaG0, sdeltaG0 = self.sdeltaG0, logger = logger)
        

        if exclude_exchanges:
            excluded_reactions = exchange_reactions + essential_reactions
        else:
            excluded_reactions = essential_reactions

        # 2
        cs_name_to_id = {}
        fp_idx = test_df['Pred outcome']=='FP'
        fp_essentials_dict = {}
        for i, row in test_df.loc[fp_idx,:].iterrows():
            cs_id = row['Carbon source ID']
            cs = row['Carbon source']
            cs_name_to_id[cs]=cs_id
            logger.info(f'Finding essential reactions for {cs_id} (FP)')
            constraints={f'R_EX_{cs_id}_e':uptake_rate}
            if auxotrophy_constraints:
                constraints.update({key: value for key, value in auxotrophy_constraints.items() if not constraints.get(key)})
            
            curr_essentials = reframed.essential_reactions(gf_model, min_growth=min_growth, constraints=constraints)
            fp_essentials_dict[cs] = [x for x in curr_essentials if not x in excluded_reactions]
            logger.info('Unique essential reactions: %s', fp_essentials_dict[cs])


        # Summarize data from 2
        all_fp_essentials = []
        rxn_count = defaultdict(int)
        no_solution = []
        rxn_to_cs_dict = defaultdict(list)
        for cs, essential_rxns_i in fp_essentials_dict.items():
            if len(essential_rxns_i):
                all_fp_essentials += essential_rxns_i
                for rxn in essential_rxns_i:
                    rxn_count[rxn] += 1
                    rxn_to_cs_dict[rxn].append(cs)
            else:
                no_solution.append(cs)

        # 3. if any reactions show up in more than two sets
        reactions_to_remove = []
        solved = []
        for key, value in rxn_count.items():
            if value > 1:
                reactions_to_remove.append(key)
                for cs in rxn_to_cs_dict[key]:
                    logger.info(f'Solved {cs} by removing {key}')
                    solved.append(cs)
        logger.info(fp_essentials_dict)

        # 4-6. look for transport reactions
        # if not, check if a transport reaction transporting the actual metabolite is the list from 2. Remove this one
        for cs, essential_rxns_i in fp_essentials_dict.items():
            if cs in solved or cs in no_solution:
                continue
            else:
                cs_id = cs_name_to_id[cs]
                scores = []
                for r_id in essential_rxns_i:
                    # Is transport reactions
                    r = gf_model.reactions[r_id]
                    is_transport = is_transport_reaction(r, gf_model)
                    r_metabolites = [m_id[2:-2] for m_id in r.stoichiometry.keys()]
                    contains_cs = cs_id in r_metabolites
                    scores.append(1*is_transport + 2*contains_cs)
                rmv_r_id = essential_rxns_i[np.argmax(scores)]
                reactions_to_remove.append(rmv_r_id)
                for cs in rxn_to_cs_dict[rmv_r_id]:
                    solved.append(cs)
                    logger.info(f'Solved {cs} by removing {rmv_r_id}')

        # Last restort. If there is no solution, it is likely that there are multiple transport reactions (and hence none is essential)
        # Or potentially "extracellular" reactions 
        for cs in no_solution:
            logger.info(f'Trying to solve last resort FP {cs}')
            cs_id = cs_name_to_id[cs]
            cs_reactions = gf_model.get_metabolite_reactions(f'M_{cs_id}_e')
            cs_reactions_to_remove = []
            for r_id in cs_reactions:
                is_exchange = gf_model.reactions[r_id].reaction_type == reframed.ReactionType.EXCHANGE
                if not is_exchange:
                    cs_reactions_to_remove.append(r_id)
            
            logger.info('Removing reactions: %s', cs_reactions_to_remove)
            reactions_to_remove += cs_reactions_to_remove
        return reactions_to_remove



    def relax_universe(self, tfa_solution_threshold_ratio = 0.5, epsilon = 1e-4):
        """
        Make sure that the universe can grow (as predicted by TFA) on all carbon sources with positive growth
        ToDo: Implement compartment-specific pH and metabolite concentrations ranges
        """
        logger = logging.getLogger('universe')
        logger.info('Relaxing universe...')

        cs_slack_dict = {}
        all_slacks = []
        model = self.universe.copy()

        # Run TFA on default CS to test and add constraints
        constraints = {self.test_universe_growth_reaction: self.default_cs_lb}
        growth_objective = model.get_objective()
        tfa_sol, tfa_solver, lhs_dict = self.TFA(model, constraints, logger)

        for c_source in self.any_growth_carbon_sources:
            logger.debug(c_source)
            r_ex_id = f'R_EX_{c_source}_e'
            constraints = {r_ex_id: self.default_cs_lb}
            fba_solution = reframed.FBA(model, constraints = constraints)
            
            if not fba_solution or fba_solution.fobj < 1e-2:
                logger.info(f"Universe can't grow on {c_source} (FBA). FBA solution: {fba_solution}")
                continue
            tfa_solution_threshold = tfa_solution_threshold_ratio*fba_solution.fobj

            tfa_sol = tfa_solver.solve(growth_objective, minimize=False, constraints=constraints, get_values=False)

            slack_dict = {}
            if (tfa_sol.status != fba_solution.status) or (tfa_sol.fobj < tfa_solution_threshold): 
                # Solution is not optimal nor above threshold
                
                # Find neccessary slack variables
                slack_objective = add_TFA_slack(self.deltaG0, tfa_solver, lhs_dict.copy(), tfa_solution_threshold)
                slack_solution = tfa_solver.solve(slack_objective, minimize=True, constraints=constraints, get_values=True)
                
                
                for r_id, _ in TFA_dG0.items():
                    if slack_solution.values['pos_slack_'+r_id] >1e-9:
                        logger.info(r_id, 'pos slack: ', slack_solution.values['pos_slack_'+r_id])
                        slack_dict[r_id] = slack_solution.values['pos_slack_'+r_id]
                    if slack_solution.values['neg_slack_'+r_id] >1e-9:
                        logger.info(r_id, 'neg slack: ', slack_solution.values['neg_slack_'+r_id])   
                        slack_dict[r_id] = -slack_solution.values['neg_slack_'+r_id]
                
                # Check that the slack works
                for key, value in slack_dict.items():
                    var = tfa_solver.problem.getVarByName('dG0_'+key)
                    var.LB = var.LB + value-epsilon
                    var.UB = var.UB + value+epsilon

                remove_TFA_slack(TFA_dG0, tfa_solver, lhs_dict.copy())

                
                new_tfa_solution = tfa_solver.solve(growth_objective, minimize=False, constraints=constraints, get_values=True)#model.get_objective()
                # print(new_tfa_solution)
                cs_slack_dict[c_source] = slack_dict
                logger.info(f'Slack added: {c_source}, FBA: {fba_solution.fobj}, TFA: {new_tfa_solution.fobj}')
                for key, _ in slack_dict.items():
                    if not key in all_slacks:
                        all_slacks.append(key)
            else:   
                cs_slack_dict[c_source] = None
                logger.info(f'No slack added for {c_source}')
        logger.info(f'Slack dict: %s', slack_dict)
        # universe.reactions[f'R_EX_{c_source}_e'].lb = 0
        self.universe_slack_dict = cs_slack_dict
        self.universe_all_slacks = all_slacks
        return cs_slack_dict, all_slacks


    def TFA(self, model, constraints, logger = None, objective = None, solver = None):
        if not logger:
            logger = logging.getLogger('TFA')

        absmax_dG0 = np.max(np.abs(list(self.deltaG0.values())))
        if absmax_dG0 > self.bigM:
            logger.warning(f'bigM {self.bigM} is lower than max dG values ({absmax_dG0})')

        if not solver:
            solver = solver_instance(model)
            self.init_solver(solver)

        lhs_dict = _add_TFA_constraints(model, solver, self.deltaG0, self.sdeltaG0, 
                            concentration_min=self.min_concentration,
                            concentration_max=self.max_concentration, bigM = self.bigM)

        if not objective:
            objective = model.get_objective()
        solution = solver.solve(objective, minimize=False, constraints=constraints, get_values=True)
        return solution, solver, lhs_dict

def add_TFA_slack(TFA_dG0, solver, lhs_dict, growth_lb):
    objective = {}
    for r_id, _ in TFA_dG0.items():
        solver.add_variable('pos_slack_'+r_id, lb = 0, update=False)
        solver.add_variable('neg_slack_'+r_id, lb = 0, update=False)
        solver.remove_constraint('dGsum_' + r_id)
        objective['pos_slack_'+r_id] = 1
        objective['neg_slack_'+r_id] = 1
    solver.update()
    for r_id, _ in TFA_dG0.items():
        lhs = lhs_dict[r_id]
        lhs.update({'pos_slack_'+r_id: 1, 'neg_slack_'+r_id:-1})
        solver.add_constraint('dGsum_' + r_id, lhs, '=', 0, update=False)
    
    var = solver.problem.getVarByName('Growth')
    var.LB = growth_lb
    solver.update()
    return objective

def remove_TFA_slack(TFA_dG0, solver, lhs_dict):
    remove_slacks = []
    for r_id, _ in TFA_dG0.items():
        remove_slacks.append('pos_slack_'+r_id)
        remove_slacks.append('neg_slack_'+r_id)
        solver.remove_constraint('dGsum_' + r_id)
    solver.remove_variables(remove_slacks)
    
    solver.update()
    for r_id, _ in TFA_dG0.items():
        lhs = lhs_dict[r_id]
        # lhs.update({'pos_slack_'+r_id: 1, 'neg_slack_'+r_id:-1})
        try:
            solver.add_constraint('dGsum_' + r_id, lhs, '=', 0, update=False)
        except TypeError:
            print(r_id, lhs)
    solver.update()


def truncate_deltaG0_values(deltaG0, absmax = 999, logger = None):
    if logger:
        logger.info(f"Truncate dG0 values to {absmax}")
    dg0_lim = {}
    for key, value in deltaG0.items():
        if value < 0:
            dg0_lim[key] = max(value, -absmax)
        else:
            dg0_lim[key] = min(value, absmax)
    return dg0_lim

# Get delta Gs from equilibrator
def get_dG_for_model_reactions_from_eq(model, compartment_data, null_error_override = 8.368):
    """
    # pytfa uses 2kcal/mol as the null error override -> 2*4.184
    Reframe flavor model expected

    # ToDo many reactions have 3 compartments ('e', 'c', 'p'). 
    As long as there is no potential or pH difference between 
    e and p, this can be treated as 1 compartment for thermodynamic purposes
    """
    logger = logging.getLogger('equilibrator')
    cc = ComponentContribution()
    dG_prime_std_dict = {}
    dG_prime_phys_dict = {}
    # except_reactions = [r.id for r in model.reactions.values() if len(r.stoichiometry) ==1]
    except_reactions = model.get_exchange_reactions()
    except_reactions += ['R_ATPM', 'Growth']
    for r in model.reactions.values():
        if r.id in except_reactions:
            continue
        r_compartments = get_reaction_compartments(r, model)
        r_metabolites = get_reaction_metabolites(r, model)
        if len(r_compartments) == 1:
            r_comp_data = compartment_data[r_compartments[0]]
            cc.p_H = Q_(r_comp_data['pH'])
            cc.ionic_strength = Q_("{0} mM".format(r_comp_data['ionicStr']*1e3))
            reaction_string = eq_met_dict_to_string(r_metabolites)
            # print(r.id, reaction_string)
            try:
                req = cc.parse_reaction_formula(reaction_string)
            except Exception as e:
                pass
            else:
                if req.is_balanced():
                    try:
                        dG_prime_std_dict[r.id] = cc.standard_dg_prime(req)
                        dG_prime_phys_dict[r.id] = cc.physiological_dg_prime(req)
                    except:
                        pass
                else:
                    pass
                    
        elif len(r_compartments) == 2:
            # Transport reaction, more complicated
            if 'c' in r_compartments:
                inner_comp = 'c'
            else:
                inner_comp = 'p'    
            outer_comp = [x for x in r_compartments if not x in inner_comp][0]
            
            comp_r_string = {}
            for compartment in r_compartments:
                comp_mets = get_reaction_metabolites_in_compartment(r, compartment, model)
                comp_r_string[compartment] = eq_met_dict_to_string(comp_mets)

            cc.p_h = Q_(compartment_data[inner_comp]['pH'])
            cc.ionic_strength = Q_("{0} mM".format(compartment_data[inner_comp]['ionicStr']*1e3))
            e_potential_difference = Q_("{0} mV".format(compartment_data[inner_comp]['membranePot'][outer_comp]))
            try:
                inner_rxn = cc.parse_reaction_formula(comp_r_string[inner_comp])
                outer_rxn = cc.parse_reaction_formula(comp_r_string[outer_comp])
            except Exception as e:
                continue
            else:
                outer_pH = Q_(compartment_data[outer_comp]['pH'])
                outer_ionic_strength = Q_("{0} mM".format(compartment_data[outer_comp]['ionicStr']*1e3))
                try:
                    standard_dg_prime = cc.multicompartmental_standard_dg_prime(inner_rxn, outer_rxn,
                                            e_potential_difference=e_potential_difference,
                                            p_h_outer=outer_pH,
                                            ionic_strength_outer=outer_ionic_strength)
                except Exception as e:
                    pass
                else:
                    dG_prime_std_dict[r.id] = standard_dg_prime
        else:
            logger.info(f"Can't estimate deltaG0 for {r.id} because it covers 3 compartments!")

    TFA_dG0 = {}
    TFA_dGerr = {}
    for r_id, x in dG_prime_std_dict.items():
        if not np.isfinite(x.value):
            continue
            
        TFA_dG0[r_id] = x.magnitude.nominal_value
        if (x.error == 0) or np.isnan(x.error):
            TFA_dGerr[r_id] = null_error_override
        else:
            TFA_dGerr[r_id] = x.error.magnitude

    return TFA_dG0, TFA_dGerr

# def select_consensus_reactions(alt_solution_dict, ratio = 0.5):
#     """
#     Method used to select "consensus" reactions, i.e. subsets of reactions that are in the suggested list of 
#     added reactions for more than a fraction(ratio) of all tested carbon sources.
#     """
#     rdf = get_subset_solutions(alt_solution_dict)
#     idx = (rdf['Ratio of CS']>ratio) #& (rdf['L']>1)
#     added_reactions = []
#     while sum(idx) > 0:
#         # Get consensus reactions

#         # sol = rdf.loc[idx, :].sort_values(by = ['L', 'N unique'], ascending=[True, False]).iloc[0]['Key']
#         # Choose the "longest" subset of consensus reactions above a threshold
#         sol = rdf.loc[idx, :].sort_values(by = ['L', 'N unique'], ascending=[False, False]).iloc[0]['Key']
#         added_reactions += sol.split(',')
#         reduced_solution_dict, solved_cs = get_reduced_solution_dict(alt_solution_dict, added_reactions)
#         rdf = get_subset_solutions(reduced_solution_dict)
#         idx = (rdf['Ratio of CS']>ratio) #& (rdf['L']>1)
#     # No more consensus subsets
#     return reduced_solution_dict, added_reactions

def select_consensus_reactions(alt_solution_dict, ratio = 0.5, max_exact_subset_length = 15, logger = None):
    """
    Method used to select "consensus" reactions, i.e. subsets of reactions that are in the suggested list of 
    added reactions for more than a fraction(ratio) of all tested carbon sources.
    """
    # Get size of the solutions
    lengths = []
    for m_id, lst_of_lst in alt_solution_dict.items():
        added = defaultdict(int)
        for lst in lst_of_lst:
            lengths.append(len(lst))
    max_length = np.max(lengths)

    if max_length < max_exact_subset_length:
        if logger:
            logger.info('Finding exact subset solutions')
        rdf = get_subset_solutions(alt_solution_dict, min_length=2)
        idx = (rdf['Ratio of CS']>ratio) #& (rdf['L']>1)
        # Choose the "longest" subset of consensus reactions above a threshold
        if np.sum(idx)>0:
            sol = rdf.loc[idx, :].sort_values(by = ['L', 'N unique', 'N'], ascending=[False, False, False]).iloc[0]['Key']
            added_reactions = sol.split(',')
        else:
            added_reactions = []
        # rdf = get_subset_solutions(reduced_solution_dict)
    else:
        if logger:
            logger.info('Finding pseudo subset solutions')
        # Can't use the method above because it takes too much time to compute the exact subsets
        # therefore, it here chooses the reactions that occur most frequently
        rdf = get_pseudo_subset_solutions(alt_solution_dict)
        idx = (rdf['Ratio of CS']>ratio)
        added_reactions = []
        while np.sum(idx)>0:
            sol = rdf.loc[idx, :].sort_values(by = ['N unique', 'N'], ascending=[False, False]).iloc[0]['r_id']
            added_reactions += sol
            rdf = remove_solutions_not_containing_these_reactions(alt_solution_dict, added_reactions)
            idx = (rdf['Ratio of CS']>ratio)
    if len(added_reactions):
        reduced_solution_dict, solved_cs = get_reduced_solution_dict(alt_solution_dict, added_reactions)
    else:
        reduced_solution_dict = alt_solution_dict

    # idx = (rdf['Ratio of CS']>ratio) #& (rdf['L']>1)
    # No more consensus subsets
    return reduced_solution_dict, added_reactions

def select_specific_gapfill_solutions(alt_solution_dict, reduced_solution_dict, added_reactions):
    while len(reduced_solution_dict):
        df = make_gapfill_solutions_df(reduced_solution_dict)
        try:
            sol = df.sort_values(by = ['N unique', 'L'], ascending=[False, True]).iloc[0]['Key']
        except IndexError:
            reduced_solution_dict = {}
            solved_cs = []
        else:
            added_reactions += sol.split(',')
            reduced_solution_dict, solved_cs = get_reduced_solution_dict(alt_solution_dict, added_reactions)
    return added_reactions, solved_cs

def remove_solutions_not_containing_these_reactions(alt_solution_dict, added_reactions):
    reduced_solution_dict = {}
    for m_id, lst_of_lst in alt_solution_dict.items():
        
        full_solution_set = set([r_id for lst in lst_of_lst for r_id in lst])
        if set(added_reactions).issubset(full_solution_set):
            keep_solutions = []
            for lst in lst_of_lst:
                if set(added_reactions).issubset(lst):
                    keep_solutions.append(lst)
        else:
            keep_solutions = lst_of_lst
        reduced_solution_dict[m_id] = keep_solutions
    df = get_pseudo_subset_solutions(reduced_solution_dict)
    return df



def get_pseudo_subset_solutions(alt_solution_dict):
    data = []
    unique_counter = defaultdict(int)
    counter = defaultdict(int)
    for m_id, lst_of_lst in alt_solution_dict.items():
        added = defaultdict(int)
        for lst in lst_of_lst:
            for r_id in lst:
                if added[r_id] == 0:
                    unique_counter[r_id] += 1
                    added[r_id] = 1

                counter[r_id] += 1
    data = []
    for r_id in unique_counter.keys():
        ratio = unique_counter[r_id]/len(alt_solution_dict)
        data.append([r_id, counter[r_id], unique_counter[r_id], ratio])
    df = pd.DataFrame(data, columns = ['r_id', 'N', 'N unique', 'Ratio of CS'])
    return df





def get_subset_solutions(alt_solution_dict, min_length = 1, max_length = None):
    """
    From a dictionary of solutions, generate all possible subsets and create a dataframe for
    the results. 
    """
    

    # min_length = L_max
    subset_solutions = defaultdict(int)
    subset_solutions_unique = defaultdict(int)
    for m_id, lst_of_lst in alt_solution_dict.items():
        added = defaultdict(int)
        for lst in lst_of_lst:
            for subset in powerset(lst, min_length, max_length):
                if len(subset)>= min_length:
                    key = ",".join(sorted(subset))
                    subset_solutions[key]+=1
                    if not added[key]:
                        subset_solutions_unique[key]+=1
                        added[key] = 1
    data = []
    for key in subset_solutions.keys():
        l = len(key.split(','))
        ratio = subset_solutions_unique[key]/len(alt_solution_dict)
        data.append([key, subset_solutions[key], subset_solutions_unique[key], l, ratio])
    df = pd.DataFrame(data, columns = ['Key', 'N', 'N unique', 'L', 'Ratio of CS'])
    return df

def make_gapfill_solutions_df(alt_solution_dict):
    data = []
    uniqe_solutions, _ = get_unique_gapfill_solutions(alt_solution_dict)
    for key, lst_of_lst in alt_solution_dict.items():
        for lst in lst_of_lst:
            if len(lst):
                data.append([key, ','.join(sorted(lst)), len(lst), uniqe_solutions[';'.join(lst)]])
    df = pd.DataFrame(data, columns = ['CS', 'Key', 'L', 'N unique'])
    return df

def get_unique_gapfill_solutions(alt_solution_dict):
    all_unique_solutions = defaultdict(int)
    r_occurences_cs = defaultdict(int)
    for key, list_of_solutions, in alt_solution_dict.items():
        all_cs_reactions = []
        for lst in list_of_solutions:
            # all_solutions.append(value)
            key = ','.join(sorted(lst))
            all_unique_solutions[key] += 1
            all_cs_reactions += lst
        for r_id in list(set(all_cs_reactions)):
            r_occurences_cs[r_id]+=1
    return all_unique_solutions, r_occurences_cs

def get_reduced_solution_dict(alt_solution_dict, added_reactions):
    reduced_solution_dict = {}
    solved_cs = []
    for m_id, lst_of_lst in alt_solution_dict.items():
        solved = False
        new_lst_of_lst = []
        for lst in lst_of_lst:
            new_lst = [x for x in lst if not x in added_reactions]
            if not len(new_lst):
                solved = True
                break
            else:
                if not new_lst in new_lst_of_lst:
                    new_lst_of_lst.append(new_lst)
        if solved:
            solved_cs.append(m_id)
        else:
            reduced_solution_dict[m_id] = new_lst_of_lst
    return reduced_solution_dict, solved_cs


def get_all_essential_reactions(species_abbr, model, cs_id_list, uptake_rate = -10, additional_constraints = None,  min_growth = 0.1, 
                            method = 'FBA', deltaG0 = None, sdeltaG0 = None, logger = None):

    fn = TMP_FOLDER / f'essential_reactions_{species_abbr}.json'
    if not logger:
        logger = logging.getLogger(f'essential_reactions.{species_abbr}')
    try:
        with open(fn, 'r') as f:
            save_dict = json.load(f)
    except FileNotFoundError:
        loaded_essentials = False
    else:
        loaded_essentials = True
        logger.info(f'Loaded essential reactions for {species_abbr} from %s', fn)
        essential_reactions = save_dict['essential_reactions'] 
        essential_reactions_dict = save_dict['essential_reactions_dict'] 

    if not loaded_essentials:
        # 1. get essential reactions
        essential_reactions = []
        essential_reactions_dict = {}
        for cs_id in cs_id_list:
            
            constraints={f'R_EX_{cs_id}_e':uptake_rate}
            if additional_constraints:
                constraints.update({key: value for key, value in additional_constraints.items() if not constraints.get(key)})
                
            essential_rxns_i = get_essential_reactions(species_abbr, model, min_growth, constraints, method, deltaG0=deltaG0, sdeltaG0=sdeltaG0, logger = logger)
            essential_reactions += essential_rxns_i
            essential_reactions_dict[cs_id] = essential_rxns_i

        save_dict = {}
        save_dict['essential_reactions'] = essential_reactions
        save_dict['essential_reactions_dict'] = essential_reactions_dict
        with open(fn, 'w') as f:
            json.dump(save_dict, f)

    return essential_reactions, essential_reactions_dict

def get_essential_reactions(species_abbr, model, min_growth, constraints, method = 'FBA', deltaG0 = None, sdeltaG0=None, run_FBA_first = True, 
                            ignore_model_bounds = False, logger = None):
    if not logger:
        logger = logging.getLogger(f'essential_reactions.{species_abbr}')

    if (method == 'TFA') and deltaG0 and sdeltaG0:
        logger.info(f'Finding essential reactions for growth on {", ".join(list(constraints.keys()))} using TFA')
        essential_rxns_i = TFA_essential_reactions(model, min_growth, constraints, deltaG0, sdeltaG0, 
                                                    run_FBA_first = True, ignore_model_bounds = False, logger = logger)
    else:
        logger.info(f'Finding essential reactions for growth on {", ".join(list(constraints.keys()))} using FBA')
        essential_rxns_i = reframed.essential_reactions(model, min_growth=min_growth, constraints=constraints)
    return essential_rxns_i


def _prep_for_gapfill(model, universe, ignore_model_bounds = False, deltaG0 = None):
    new_reactions = set(universe.reactions) - set(model.reactions)
    merged_model = merge_models(model, universe, inplace = False, tag='TFAgapfill')

    # Ensure that no exchange reactions from the universe are open
    for r_id in new_reactions:
        if r_id.startswith('R_EX'):
            merged_model.set_flux_bounds(r_id, lb=0)
    
    if ignore_model_bounds and deltaG0:
        print('Ignore model bounds in TFA!')
        for r_id in merged_model.reactions:
            if r_id in deltaG0:
                merged_model.set_flux_bounds(r_id, -bigM, bigM)
    return merged_model, new_reactions


def _add_TFA_constraints(model, solver, deltaG0, sdeltaG0=None, concentration_max=1e-2,
                        concentration_min=1e-5, 
                        excluded=None, temperature=298.15, bigM = 1e3):
    """
    I think the number of constraints ca be reduced by taking into account that many reactions are irreversible
    """

    curr_absmax_dG0 = np.max(np.abs(list(deltaG0.values())))
    if  curr_absmax_dG0 > bigM:
        print(f'Warning: bigM {bigM} is lower than max dG values ({curr_absmax_dG0})')

    ln_min = np.log(concentration_min)
    ln_max = np.log(concentration_max)

    R = 0.00831
    RT = temperature * R

    if not sdeltaG0:
        sdeltaG0 = {r_id: 0 for r_id in deltaG0}

    if not excluded:
        excluded = []

    included = []
    
    if not hasattr(solver, 'tFBA_flag'):
        lhs_dict = {}
        solver.tFBA_flag = True
        for r_id in model.reactions:
            if r_id in deltaG0:
                solver.add_variable('y_' + r_id, 0, 1, vartype=VarType.BINARY, update=False)
                solver.add_variable('dG_' + r_id, update=False)
                dG0_min, dG0_max = deltaG0[r_id] - sdeltaG0[r_id], deltaG0[r_id] + sdeltaG0[r_id]
                solver.add_variable('dG0_' + r_id, dG0_min, dG0_max, update=False)

                for m_id in model.reactions[r_id].stoichiometry:
                    if m_id not in excluded and m_id not in included:
                        solver.add_variable('ln_' + m_id, ln_min, ln_max, update=False)
                        included.append(m_id)

        solver.update()

        for r_id in model.reactions:
            if r_id in deltaG0:
                # If v_i > 0 -> y = 0, if v_i < 0 -> y = 1
                solver.add_constraint('lb_' + r_id, {r_id: 1, 'y_' + r_id: bigM}, '>', 0, update=False)
                solver.add_constraint('ub_' + r_id, {r_id: 1, 'y_' + r_id: bigM}, '<', bigM, update=False)
                
                # If v_i > 0 -> y = 0 -> dG < 0, if v_i < 0 -> y = 1 -> dG > 0
                solver.add_constraint('lb_dG_' + r_id, {'dG_' + r_id: -1, 'y_' + r_id: bigM}, '>', 0,
                                      update=False)
                solver.add_constraint('ub_dG_' + r_id, {'dG_' + r_id: -1, 'y_' + r_id: bigM}, '<', bigM,
                                      update=False)
                lhs = {'ln_' + m_id: RT * coeff for m_id, coeff in model.reactions[r_id].stoichiometry.items()
                       if m_id in included}
                lhs.update({'dG0_' + r_id: 1, 'dG_' + r_id: -1})
                solver.add_constraint('dGsum_' + r_id, lhs, '=', 0, update=False)
                lhs_dict[r_id] = lhs
        solver.update()
        return lhs_dict
    else:
        return None
def _add_gapfilling_constraints(model, solver, new_reactions, min_growth, bigM = 1e3, 
                                use_indicator_constraints = False, logger = None):
    if not logger:
        logger = logging.getLogger('gapfill')

    if not hasattr(solver, '_gapfill_flag'):
        solver._gapfill_flag = True

        for r_id in new_reactions:
            solver.add_variable('z_' + r_id, 0, 1, vartype=VarType.BINARY)#, update=False)

        solver.update()
        if use_indicator_constraints:
            logger.warning(' Use indicator constraints - only with gurobi')
            for r_id in new_reactions:
                var_flux = solver.problem.getVarByName(r_id)
                var_usage = solver.problem.getVarByName('z_'+r_id)
                solver.problem.addConstr((var_usage < 0.99) >> (var_flux == 0), name = 'ind_'+r_id)
        else:  
            for r_id in new_reactions:
                solver.add_constraint('lbz_' + r_id, {r_id: 1, 'z_'+r_id: bigM}, '>', 0)#, update=False)
                solver.add_constraint('ubz_' + r_id, {r_id: 1, 'z_'+r_id: -bigM}, '<',0)#, update=False)
        biomass = model.biomass_reaction
        solver.add_constraint('min_growth', {biomass: 1}, '>', min_growth)#, update=False)

        solver.update()
    else:
        logger.info('Solver already has gapfilling constraints')

def _get_added_gf_reactions(solution, new_reactions, fluxtol = 1e-9, raise_error = False, tag = '', logger = None):
    print(f'Solution status: {solution.status}')
    if solution.status == Status.OPTIMAL:
    
        inactive = [r_id for r_id in new_reactions if abs(solution.values[r_id]) < fluxtol]
        added_reactions = [r_id for r_id in new_reactions if abs(solution.values[r_id]) >= fluxtol]
        added_reactions.sort()
        failed = False
    else:
        err_message = 'Failed to gapfill model for medium {}'.format(tag)
        if raise_error:
            raise RuntimeError(err_message)
        else:
            if logger:
                logger.warning(err_message)
                # print(solution, new_reactions)
        failed = True
        inactive = None
        added_reactions = None
    return inactive, added_reactions, failed


def rank_auxotrophy_solutions(model, potential_solutions, auxotrophy_met_id):
    scores_dict = {}
    met_reactions = model.get_metabolite_reactions(f"M_{auxotrophy_met_id}_c")#_e?
    for r_id in potential_solutions:
        r = model.reactions[r_id]
        if r.reaction_type == reframed.ReactionType.EXCHANGE:
            # Remove exchanges doesn't make sense
            continue
        score = 0
        # Prioritize to remove reactions without gene annotation
        # The scores are chosen so all sums are unique
        if not r.gpr:
            score += 4
        if r_id in met_reactions:
            score += 2
        if r.reaction_type == reframed.ReactionType.ENZYMATIC:
            score += 1
        scores_dict[r_id] = score
    return scores_dict





if __name__ == '__main__':
    repo_path = Path('/Users/snorre/git/mwf_gems')
    carveme_draft_folder = repo_path / 'models/carveme'
    binary_growth_data_path = repo_path / 'data'/'growth_no_growth.csv'
    carbon_source_ids_path = repo_path / 'data'/'carbon_source_ids_curated.csv'

    gapfilling_data_folder = repo_path / 'gapfilling_data'
    M9_minimal_media_file = gapfilling_data_folder / 'M9_minimal_media_bigg.csv'
    # vitamins_file = gapfilling_data_folder / 'vitamins_bigg.csv'
    bigg_universe_fn = gapfilling_data_folder / 'universe_bacteria.xml'#'bigg_universe.xml'
    compartment_data_fn = gapfilling_data_folder /'compartment_data.json'


    N = niceGAME(bigg_universe_fn, M9_minimal_media_file, binary_growth_data_path, carbon_source_ids_path, compartment_data_fn)
    N.get_universe_reactions_delta_G_from_equilibrator()
    # cs_slack_dict, all_slacks = N.relax_universe()
    N.set_model_folder(carveme_draft_folder)
    auxotrophy_dict = {
    'Ml': {'amino acids': ['pro__L', 'cys__L'], 'vitamins':['thm', 'btn']},
    'Oa': {'vitamins': ['thm']}
    }
    N.set_auxotrophy_dict(auxotrophy_dict)
    N.load_gapfill_solutions()
    for species_abbr in ['At', 'Ct', 'Ml', 'Oa']:#, ,, 'At', 'Ct', 
        N.load_model(species_abbr)
        if not N.gapfill_solutions.get(species_abbr):
            N.gapfill_model_on_all_cs(species_abbr, N_alternative_solutions = 10, add_TFA=False)
        N.store_gapfill_solutions()
        # N.load_gapfill_solutions()
        N.select_gapfill_solutions_and_gapfill(species_abbr)
        N.check_auxotrophies(species_abbr)
        N.save_gf_model(species_abbr, simulation_ready = True)



