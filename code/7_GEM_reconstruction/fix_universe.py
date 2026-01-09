# -*- coding: utf-8 -*-
"""
module: fix_universe

description: functions used to fix the carveme universe

author: Snorre Sulheim
date: October 27, 2023

"""

import numpy as np
import reframed
from ng_utils import get_mol_weight, check_reaction_mass_balance
from pathlib import Path
import logging

HOME_PATH = Path.home()
REPO_PATH =  HOME_PATH / 'git' / 'coexistence' #Path(dotenv.find_dotenv()).parent
ECOLI_FN =  REPO_PATH / "data" / "7_GEM_reconstruction" / 'iML1515.xml'


def add_biotin_synthase_reactions(model, logger = None):
    """
    The CarveMe bacteria universe can't synthesize biotin. To fix this issue I add all reactions that 
    are essential for biotin synthesis in E.coli iML1515
    
    > ecoli_fn = gapfilling_data_folder / 'iML1515.xml'
    > ecoli = reframed.load_cbmodel(ecoli_fn)
    > essential_reactions_ecoli_wt = reframed.essential_reactions(ecoli)
    > essential_reactions_ecoli_btn = reframed.essential_reactions(ecoli, constraints={'R_EX_btn_e':(-0.01, 0)})
    > diff = [x for x in essential_reactions_ecoli_wt if not x in essential_reactions_ecoli_btn]

    """
    btn_synthesis_essential_reactions = ['R_DBTS',
                                         # 'R_DM_amob_c',
                                         'R_EX_meoh_e',
                                         'R_AMAOTr',
                                         'R_MALCOAMT',
                                         'R_OGMEACPS',
                                         'R_OGMEACPR',
                                         'R_OPMEACPS',
                                         'R_OPMEACPD',
                                         'R_EPMEACPR',
                                         'R_AOXSr2',
                                         'R_MEOHtrpp',
                                         'R_OGMEACPD',
                                         'R_EGMEACPR',
                                         'R_OPMEACPR',
                                         'R_PMEACPE',
                                         'R_MEOHtex',
                                         'R_BTS5']
    # print(ECOLI_FN, REPO_PATH, HOME_PATH)
    ecoli = reframed.load_cbmodel(ECOLI_FN)

    added_reactions = []
    for r_id in btn_synthesis_essential_reactions:
        if model.reactions.get(r_id):
            continue
        r = ecoli.reactions[r_id]
        for m_id, _ in r.stoichiometry.items():
            if not m_id in model.metabolites:
                model.metabolites[m_id] = ecoli.metabolites[m_id]
        model.reactions[r_id] = r
        added_reactions.append(r_id)

    model.metabolites['M_pimACP_c'].metadata['FORMULA'] = 'C391H613N96O145P1S3'
    model.metabolites['M_ogmeACP_c'].metadata['FORMULA'] = 'C390H609N96O146P1S3'
    model.metabolites['M_hgmeACP_c'].metadata['FORMULA'] = 'C390H611N96O146P1S3'
    model.metabolites['M_egmeACP_c'].metadata['FORMULA'] = 'C390H609N96O145P1S3'
    model.metabolites['M_gmeACP_c'].metadata['FORMULA'] = 'C390H611N96O145P1S3'
    model.metabolites['M_opmeACP_c'].metadata['FORMULA'] = 'C392H613N96O146P1S3'
    model.metabolites['M_hpmeACP_c'].metadata['FORMULA'] = 'C392H615N96O146P1S3'
    model.metabolites['M_epmeACP_c'].metadata['FORMULA'] = 'C392H613N96O145P1S3'
    model.metabolites['M_pmeACP_c'].metadata['FORMULA'] = 'C392H615N96O145P1S3'
    if logger:
        logger.info(f'Added biotin synthase reactions %s', added_reactions)


def fix_misc(model, logger = None):
    if logger:
        logger.info('Fixing miscellaneous issues in the universe model')
    
    # Exchanges for na1 and ni2 are missing
    met_ids = ['na1', 'ni2', 'pectin']
    for m_id in met_ids:
        reaction_string = f"R_EX_{m_id}_e: M_{m_id}_e <->  [0, 1000]"
        model.add_reaction_from_str(reaction_string)
        model.reactions[f'R_EX_{m_id}_e'].reaction_type = reframed.ReactionType.EXCHANGE

    # Give pectin met metadata
    model.metabolites['M_pectin_e'].metadata['CHARGE'] = '-1'
    model.metabolites['M_pectin_e'].metadata['FORMULA'] = 'C6H7O6'
    model.metabolites['M_pectin_e'].metadata['metanetx.chemical'] = 'MNXM148417'

    # Replace P0 by P in formulas
    for m in model.metabolites.values():
        try:
            formula = m.metadata['FORMULA']
        except KeyError:
            pass
        else:
            if 'P0' in formula:
                m.metadata['FORMULA'] = formula.replace('P0','P')

    # Delete GCDH (is it redundant and was previously found to cause issues with CO2)
    model.remove_reaction('R_GCDH')





def remove_metabolites_without_exchange_reactions(model, logger = None):
    """
    Remove metabolites that do not have an exchange reaction associated with them
    """
    to_remove = ['M_apc',
                 'M_cm',
                 'M_fusa',
                 'M_ttrcyc',
                 'M_cd2',
                 'M_novbcn',
                 'M_doxrbcn']

    for m_id_head in model.metabolites:
        for tail in ['_c', '_p', '_e']:
            m_id = m_id_head+tail
            if m_id in model.metabolites:
                rxns = model.get_metabolite_reactions(m_id)
                model.remove_metabolite(m_id)
                logger.info(f'Removed metabolite {m_id}')
                if len(rxns) == 0:
                    model.remove_reactions(rxns)
                    logger.info(f'Removed {rxns} related to {m_id} since it had no exchange reaction')
    