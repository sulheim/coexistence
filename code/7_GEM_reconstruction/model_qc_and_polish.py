# -*- coding: utf-8 -*-
"""
module: model_qc_and_polish

description: functions used to curate gapfilled final models

author: Snorre Sulheim
date: October 27, 2023

"""
from ng_utils import *
from collections import OrderedDict
import pandas as pd


universe_duplicated_mets =[["btoh_c", "1btol_c"],
                           ["2h3k5m_c","hkmpp_c"],
                           ["2h3opp_c","2h3oppan_c"],
                           ["2hxmp_c","2hymeph_c"],
                           ["2kmb_c","4met2obut_c"],
                           ["34dhpacet_c","34dhpha_c"],
                           ["3hbcoa_c","3hbycoa_c"],
                           ["3hibutcoa_c","hibcoa_c"],
                           ["5d4dglcr_c","5dh4dglc_c"],
                           ["aacald_c","amacald_c"],
                           ["abt__L_c","abt_c"],
                           ["abt__L_e","abt_e"],
                           ["acamoxm_c","nal2a6o_c"],
                           ["dd2coa_c","trans_dd2coa_c"],
                           ["dscl_c","shcl_c"],
                           ["eths_c","ethso3_c"],
                           ["eths_e","ethso3_e"],
                           ["galct__D_c","galctr__D_c"],
                           ["galct__D_e","galctr__D_e"],
                           ["glcn__D_c","glcn_c"],
                           ["glcn__D_e","glcn_e"],
                           ["isetac_c","istnt_c"],
                           ["isetac_e","istnt_e"],
                           ["metox_c","metsox_S__L_c"],
                           ["metox_e","metsox_S__L_e"],
                           # ["nh4_c", "nh3_c"],
                           ["orn_c", "orn__L_c"],
                           ["orn_e", "orn__L_e"],
                           ["scl_c","srch_c"],
                           ["sulfac_c", "sula_c"],
                           ["sulfac_e", "sula_e"]]

def polish_model(model, logger = None):
    # Remove intracellular metabolites that are disconnected
    remove_mets = []
    for m_id in model.metabolites:
        rxns = model.get_metabolite_reactions(m_id)
        if not len(rxns):
            remove_mets.append(m_id)
    model.remove_metabolites(remove_mets)
    if logger:
        logger.info(f'Removed these disconnected metabolites: %s', remove_mets)
    # add bigg.reaction and bigg.metabolite annotations
    for r in model.reactions.values():
        r.metadata['bigg.reaction'] = r.id.removeprefix('R_')

    for m in model.metabolites.values():
        m.metadata['bigg.metabolite'] = m.id[:-2].removeprefix('M_')



def curate_model(model, logger = None):
    # 1. find and remove duplicated metabolites
    msg = remove_duplicated_metabolites(model)
    if logger:
        logger.info(msg)

    # 2. find and remove duplivated reactions
    msg = remove_duplicated_reactions(model, logger)
    if logger:
        logger.info(msg)

    # 3. Fix unbalanced equations
    fix_unbalanced_equations(model)


    # 5. misc
    curate_misc(model)

def test_balance(model):
    # Check balance
    balance_list = []
    exchanges = model.get_exchange_reactions()
    for r in model.reactions.values():
        if (r.id in exchanges) or (r.id == 'Growth'):
            continue
        balance_r = check_reaction_balance(model, r)
        balance_list.append([r.id]+list(balance_r))
    df = pd.DataFrame(balance_list, columns = ['r_id', 'mass_balance', 'mass', 'element_balance', 'element_dict', 'charge_balance', 'charge'])
    idx = df.mass_balance & df.element_balance & df.charge_balance
    return df.loc[~idx, :]

def test_leaky(model):
    exchanges = model.get_exchange_reactions()
    for r_id in exchanges:
        model.reactions[r_id].lb = 0

    for m in model.metabolites:
        if m.id[-2:] == '_c':
            model.add_reaction_from_string('R_DM_{m.id}')
            r_id = f"R_DM_{m.id.lstrip('M_')}"
            reaction_string = f"{r_id}: {m.id} <->  [0, 1000]"
            model.add_reaction_from_string(reaction_string)
            sol = reframed.FBA(model, objective=r_id)

def curate_misc(model):
    # Replace P0 by P in formulas
    for m in model.metabolites.values():
        try:
            formula = m.metadata['FORMULA']
        except KeyError:
            pass
        else:
            if 'P0' in formula:
                m.metadata['FORMULA'] = formula.replace('P0','P')

def fix_unbalanced_equations(model):
    dic = {'R_CMCBTFR1': {'M_h_c':2},
           'R_CMCBTFR2': {'M_h_c':2},
           'R_CMCBTFU': {'M_h_c':0},
           'R_MCBTFabcpp': {'M_h_c':0},
           'R_DHBSZ3FEabcpp': {'M_h_c':0},
           'R_DTBTt': {'M_h_c':0},
           'R_SALCHS4FEabcpp': {'M_h_c':0},
           'R_TAGabc': {'M_h_c':1},
           'R_UACCpts': {'M_h_c':0},
           'R_MAN6Gpts': {'M_h_c':0}}

    if model.metabolites.get('M_fcmcbtt_c'):
        model.metabolites.M_fcmcbtt_c.metadata['FORMULA'] = 'C33FeH48N5O13'
    
    for key, stoic in dic.items():
        if model.reactions.get(key):
            model.reactions[key].stoichiometry.update(stoic)


def fix_biomass(model, biomass_id = 'Growth', 
                add_biotin = True, biotin_coeff = -2e-6, universe = None, logger = None):
    """
    Biotin is not included in carveme bacteria_universe biomass equation, but for most bacteria this is an essential vitamin
    The coefficient -2e-6 is the coefficient used in iML1515 (-0.000233 in iJO1366).
    """
    r = model.reactions[biomass_id]

    if add_biotin:
        if biotin_coeff > 0:
            raise ValueError('Biotin coefficient must be negative')
        m_id = 'M_btn_c'
        if not model.metabolites.get(m_id):
            if universe:
                model.add_metabolite(universe.metabolites[m_id])
            else:
                m_btn = reframed.Metabolite(m_id)
                m_btn.metadata['FORMULA'] = 'C10H15N2O3S'
                m_btn.metadata['CHARGE']  = '-1'
                model.add_metabolite(m_btn)
        r.stoichiometry.update({m_id: biotin_coeff})
    model.add_reaction(r)
    rescale_biomass_equation(model, biomass_id, logger = logger)

def remove_duplicated_metabolites(model):
    metabolites_to_delete = []
    for m1_id, m2_id in universe_duplicated_mets:
        if model.metabolites.get(f'M_{m1_id}') and model.metabolites.get(f'M_{m2_id}'):
            m1 = model.metabolites[f'M_{m1_id}']
            m2 = model.metabolites[f'M_{m2_id}']
            m1_formula_dict = get_element_dict(m1)
            m2_formula_dict = get_element_dict(m2)
            if m1_formula_dict == m2_formula_dict:
                if not m1.metadata.get('CHARGE'):
                    if m2.metadata.get('CHARGE'):
                        charge = m2.metadata['CHARGE']
                    else:
                        charge = None
                    m1.metadata['CHARGE'] = charge
                    
                m2_reactions = model.get_metabolite_reactions(m2.id)
                for r_id in m2_reactions:
                    r = model.reactions[r_id]
                    coeff = r.stoichiometry[m2.id]
                    r.stoichiometry[m1.id] = coeff
                    del r.stoichiometry[m2.id]
                metabolites_to_delete.append(m2.id)
    model.remove_metabolites(metabolites_to_delete)
    msg = f"Deleted these metabolites because they were duplicates: {', '.join(metabolites_to_delete)}"
    return msg


def remove_duplicated_reactions(model, logger = None):
    # Check for duplicates
    rs_dict = OrderedDict()
    for r in model.reactions.values():
        rs_dict[r.id] = dict(r.stoichiometry)

    reaction_ids_to_delete = []
    reactions_to_update = []
    protected_reactions = ['R_ATPM', 'R_NTP1']
    for i, r1 in enumerate(model.reactions.values()):
        if not rs_dict.get(r1.id):
            # Reaction has already been identified as a duplicate
            continue
        duplicates = [r_id for r_id, x in rs_dict.items() if (x == rs_dict[r1.id] and r_id != r1.id)]
        duplicates = [r_id for r_id in duplicates if not r_id in protected_reactions]
        n_duplicates = len(duplicates)
        if  n_duplicates > 0:
            # print(r1.id)
            # duplicated_reactions = [r for i, r in enumerate(model.reactions) if duplicates[i]]
            r1.metadata['Duplicates'] = ','.join(duplicates)
            if logger:
                logger.info(f"{r1.id} duplicates: {r1.metadata['Duplicates']}") 
            for r_id in duplicates:
                
                r2 = model.reactions[r_id]
                merge_reaction_metadata(r1, r2)
                r1.lb = min(r2.lb, r1.lb)
                r1.ub = max(r2.ub, r2.ub)

                # Merge GPRs
                merge_GPRs(r1, r2)
                # print(f'Merged {r2.id} into {r1.id}')
                # print(r.id, r.name, r.lb, r.ub, r.metadata, r)
                del rs_dict[r_id]
                reaction_ids_to_delete.append(r2.id)
            reactions_to_update.append(r1)
    for r in reactions_to_update:
        model.add_reaction(r)
    model.remove_reactions(reaction_ids_to_delete)
    msg = f"Deleted these reactions because they are duplicates: {', '.join(reaction_ids_to_delete)}"
    return msg

def merge_GPRs(r1, r2):
    if r1.gpr:
        if r2.gpr:
            r1.gpr.proteins = list(set(r1.gpr.proteins + r2.gpr.proteins))
        else:
            pass
    else:
        # No gpr for r1
        if r2.gpr:
            r1.gpr = r2.gpr
        else:
            #No gpr for either r1 or r2
            pass


def merge_reaction_metadata(r1, r2):
    for key, value in r2.metadata.items():
        if not isinstance(value, list):
            value = [value]
        if key == 'XMLAnnotation':
            continue
        
        if r1.metadata.get(key):
            r1_value = r1.metadata[key]
            if not isinstance(r1_value, list):
                r1_value = [r1_value]
            new_value = list(set(r1_value+value))
        else:
            new_value = value
        if key == 'SBOTerm':
            new_value = new_value[0]
        else:    
            if len(new_value) > 1:
                r1.metadata[key] = new_value
            elif len(new_value) == 1:
                r1.metadata[key] = new_value[0]
            else:
                pass

        # print(r1.id, key, new_value)

def rescale_biomass_equation(model, biomass_reaction_id = 'Growth', logger = None):
    r_bio = model.reactions[biomass_reaction_id]
    mass_balance = check_reaction_mass_balance(model, r_bio)

    if not np.isfinite(mass_balance):
        msg = "Biomass equation sums to np.nan! Can't fix this"
        if logger:
            logger.warning(msg)
            return False
        else:
            raise ValueError(msg)

    if np.abs(mass_balance-1)>1e-3:
        msg = f'Biomass equationsums to {mass_balance} -> rescaling'
        if logger:
            logger.info(msg)
        else:
            print(msg)
        # Rescale
        dic = {}
        gam = r_bio.stoichiometry['M_adp_c']
        for key, value in r_bio.stoichiometry.items():
            if key.lstrip('M_') in ['h2o_c','atp_c']:
                value_no_gam = (value + gam) # value is negative
                dic[key] = (value_no_gam/mass_balance) - gam
            elif key.lstrip('M_') in ['h_c', 'pi_c','adp_c']:
                value_no_gam = value - gam
                dic[key] = (value_no_gam/mass_balance) + gam
            else:
                dic[key] = value/mass_balance
            dic[key] = np.round(dic[key], 6)

        r_bio.stoichiometry.update(dic)
        model.add_reaction(r_bio)
        new_mass_balance = check_reaction_mass_balance(model, r_bio)

        if logger:
            logger.info(f'Rescaled biomass equation from {mass_balance} to {new_mass_balance}')
    return True