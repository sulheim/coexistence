import reframed
import xmltodict
import numpy as np
from itertools import chain, combinations
import re
from collections import defaultdict
import json
import reframed
from reframed.solvers.solution import Status
from reframed.solvers import solver_instance



def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False

def powerset(iterable, min_length = 1, max_length = None, step = 1):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    if not max_length:
        max_length = len(s)
    return chain.from_iterable(combinations(s, r) for r in range(min_length, max_length+1, step))
    

def merge_models(model1, model2, inplace=True, tag=None):
    """
    Merges two reframed models.

    Copied from carveme.reconstruction.gapfilling
    """

    if not inplace:
        model1 = model1.copy()

    for c_id, comp in model2.compartments.items():
        if c_id not in model1.compartments:
            model1.compartments[c_id] = comp

    for m_id, met in model2.metabolites.items():
        if m_id not in model1.metabolites:
            model1.metabolites[m_id] = met

    for r_id, rxn in model2.reactions.items():
        if r_id not in model1.reactions:
            model1.reactions[r_id] = rxn
            if tag:
                rxn.metadata['GAP_FILL'] = tag

    return model1

def get_reaction_compartments(r, model):
    """
    For reframed flavor models
    """
    compartments = []
    for m_id,i in r.stoichiometry.items():
        m = model.metabolites[m_id]
        compartments.append(m.compartment)
    return list(set(compartments))

def get_reaction_metabolites_in_compartment(r, compartment, model):
    comp_mets = {}
    for m_id, coeff in r.stoichiometry.items():
        m = model.metabolites[m_id]
        if m.compartment == compartment:
            comp_mets[m] = coeff
    return comp_mets

def get_reaction_metabolites(r, model):
    mets = {}
    for m_id, coeff in r.stoichiometry.items():
        m = model.metabolites[m_id]
        mets[m] = coeff
    return mets

def is_transport_reaction(r, model):
    compartments = get_reaction_compartments(r, model)
    return len(compartments) > 1

def eq_met_dict_to_string(met_dict, lstrip = 'M_'):
    """
    Converts a metabolite dict to a format that can be handled by equilibrator_api

    """
    r_strings = []
    p_strings = []
    for m, coeff in met_dict.items():
        if len(lstrip):
            m_id = m.id.lstrip(lstrip)[:-2]
        else:
            m_id = m.id[:-2]
        if coeff < 0:
            r_strings.append("{0} bigg.metabolite:{1}".format(-coeff, m_id))
        else:
            p_strings.append("{0} bigg.metabolite:{1}".format(coeff, m_id))
    r_string = " + ".join(r_strings)
    p_string = " + ".join(p_strings)
    reaction_string = r_string + ' = ' + p_string
    return reaction_string


def get_model_size(model):
    dic = {}
    for rxn_type in reframed.ReactionType:
        dic[rxn_type.value] = len(model.get_reactions_by_type(rxn_type))
    return dic

def set_base_environment(model, metabolites, max_uptake = 50):
     # Check if _e notation is used
    if metabolites[0].endswith('_e'):
        metabolites = [x.rstrip('_e') for x in metabolites]

    env = reframed.Environment.from_compounds(metabolites, max_uptake = max_uptake)
    env.apply(model)

def add_missing_exchanges(model, metabolites, ex_reaction_identifier = 'R_EX_', metabolite_id_prefix = 'M_', universe = None):
    # Check if _e notation is used
    if metabolites[0].endswith('_e'):
        metabolites = [x.rstrip('_e') for x in metabolites]
    fmt_func = lambda x: f"{ex_reaction_identifier}{x}_e"
    exchanges = map(fmt_func, metabolites)

    for r_id, m_id in zip(exchanges, metabolites):
        if not r_id in model.reactions:
            if universe and r_id in universe.reactions:
                r = universe.reactions[r_id]
                for m_id, _ in r.stoichiometry.items():
                    if not m_id in model.metabolites:
                        model.metabolites[m_id] = universe.metabolites[m_id]
                model.reactions[r_id] = r
            else:
                reaction_string = "{0}: {1}{2}_e <->  [0, 1000]".format(r_id, metabolite_id_prefix, m_id)
                model.add_reaction_from_str(reaction_string)
                model.reactions[r_id].reaction_type = reframed.ReactionType.EXCHANGE

def add_missing_exchanges_from_reactions_ids(model, reaction_ids, metabolite_id_prefix = 'M_', universe = None):
    for r_id in reaction_ids:
        if not r_id in model.reactions:
            if universe:
                r = universe.reactions[r_id]
                for m_id, _ in r.stoichiometry.items():
                    if not m_id in model.metabolites:
                        model.metabolites[m_id] = universe.metabolites[m_id]
                model.reactions[r_id] = r
            else:
                m_id = r_id.lstrip('R_EX_').rstrip('_e')
                reaction_string = "{0}: {1}{2}_e <->  [0, 1000]".format(r_id, metabolite_id_prefix, m_id)
                model.add_reaction_from_str(reaction_string)
                model.reactions[r_id].reaction_type = reframed.ReactionType.EXCHANGE
            
def add_reactions_from_universe(model, universe, reaction_id_list):
    for r_id in reaction_id_list:
        r = universe.reactions[r_id]
        for m_id, _ in r.stoichiometry.items():
            if not m_id in model.metabolites:
                model.metabolites[m_id] = universe.metabolites[m_id]
        model.reactions[r_id] = r

def fix_compartments(model):
    old_to_new_compartments = {
        'C_c':'c',
        'C_p':'p',
        'C_e':'e'
    }
    for m in model.metabolites.values():
        m.compartment = old_to_new_compartments[m.compartment]

    for comp in model.compartments.values():
        comp.id = old_to_new_compartments[comp.id]

    for old_id, new_id in old_to_new_compartments.items():
        model.compartments[new_id] = model.compartments[old_id]
        del model.compartments[old_id]

def fix_reframed_annotations(model):
    for r in model.reactions.values():
        annotation_dict = convert_xml_to_annotation_dict(r)
        r.metadata.update(annotation_dict)
        if r.metadata.get('XMLAnnotation'):
            del r.metadata['XMLAnnotation']

    for m in model.metabolites.values():
        annotation_dict = convert_xml_to_annotation_dict(m)
        m.metadata.update(annotation_dict)
        if m.metadata.get('XMLAnnotation'):
            del m.metadata['XMLAnnotation']






def convert_xml_to_annotation_dict(x):
    """
    x can be a reaction or a metabolite
    """
    
    ann_dict = {}
    try:
        ann_string = x.metadata['XMLAnnotation']
        temp_dic = xmltodict.parse(ann_string)
        entries = temp_dic['annotation']['rdf:RDF']['rdf:Description']['bqbiol:is']['rdf:Bag']['rdf:li']
    except (TypeError, KeyError) as e:
        pass
    else:
        for entry in entries:
            try:
                key = entry['@rdf:resource']
            except TypeError:
                pass
            else:
                db, value = key.split('/')[-2:]
                ann_dict[db]=value
            
    return ann_dict

# def convert_notes_to_xml_annotation(x):
#     """
#     x can be a reaction or a metabolite
#     """
    
#     ann_dict = {}
#     skip = ['CHARGE', 'FORMULA', 'SBOterm']
#     for key, values in x.notes:
#         if isinstance(values, (str, int)):
#             values = [values]
#         for value in values:

#     try:
#         ann_string = x.metadata['XMLAnnotation']
#         temp_dic = xmltodict.parse(ann_string)
#         entries = temp_dic['annotation']['rdf:RDF']['rdf:Description']['bqbiol:is']['rdf:Bag']['rdf:li']
#     except (TypeError, KeyError) as e:
#         pass
#     else:
#         for entry in entries:
#             try:
#                 key = entry['@rdf:resource']
#             except TypeError:
#                 pass
#             else:
#                 db, value = key.split('/')[-2:]
#                 ann_dict[db]=value
            
#     return ann_dict



def MCC(p):
    """
    Matthews coefficient of correlation
    p = confusion matrix
    """
    nominator = p['TP']*p['TN'] - p['FP']*p['FN']
    denominator = np.sqrt((p['TP']+p['FP'])*(p['TP']+p['FN'])*(p['TN']+p['FP'])*(p['TN']+p['FN']))

    if denominator == 0:
        return 0
    else:
        return  nominator/denominator

def precision(pmd):
    return pmd['TP'] / (pmd['FP'] + pmd['TP'])
def accuracy(pmd):
    return (pmd['TP'] + pmd['TN'])/ (pmd['TP'] + pmd['FN'] + pmd['TN'] + pmd['FP'])
def recall(pmd):
    return pmd['TP'] / (pmd['FN'] + pmd['TP'])
def f1_score(pmd):
    p = precision(pmd)
    r = recall(pmd)
    return 2*p*r/(p+r)



def check_reaction_balance(model, r):
    # Mass balance
    mass = check_reaction_mass_balance(model, r)
    if np.abs(mass) < 1e-6:
        mass_balance = True
    else:
        mass_balance = False

    # Element balance
    element_balance, element_dict = check_reaction_element_balance(model, r)

    # Charge balance
    charge_balance, charge = check_charge_balance(model, r)

    return mass_balance, mass, element_balance, element_dict, charge_balance, charge

def check_charge_balance(model, r):
    total_charge = 0

    for key, coeff in r.stoichiometry.items():
        m = model.metabolites[key]
        
        try:
            charge = m.metadata['CHARGE']
        except KeyError:
            charge = np.nan

        total_charge += float(charge)*coeff

    if total_charge == 0:
        is_balanced = True
    else:
        is_balanced = False
    return is_balanced, total_charge



def check_reaction_element_balance(model, r):
    element_dict = defaultdict(int)
    for key, coeff in r.stoichiometry.items():
        m = model.metabolites[key]
        m_element_dict = get_element_dict(m)
        for e, n in m_element_dict.items():
            element_dict[e] += n*coeff
    if sum(element_dict.values()) == 0:
        is_balanced = True
    else:
        is_balanced = False
    return is_balanced, element_dict

def check_reaction_mass_balance(model, r):
    total_sum = 0
    for key, value in r.stoichiometry.items():
        met_weight = get_mol_weight(model.metabolites[key])
        total_sum += -value*met_weight*1e-3
    return total_sum

def get_element_dict(metabolite):
    try:
        formula = metabolite.metadata['FORMULA']
    except KeyError:
        return np.nan
    else:
        element_count = extract_elements_and_counts(formula)
        return element_count

def get_mol_weight(metabolite):
    element_count = get_element_dict(metabolite)
    if not isinstance(element_count, dict):
        return np.nan
    else:
        total_mass = 0
        for key, value in element_count.items():
            try:    
                total_mass += elements_and_molecular_weights[key]*value
            except KeyError:
                total_mass = np.nan
                break
        return total_mass

def extract_elements_and_counts(formula):
    element_pattern = r'([A-Z][a-z]*)(\d*)'
    # element_pattern = r'([A-Z][a-z]*)(\d*)'
    elements = re.findall(element_pattern, formula)
    element_count = defaultdict(int)
    
    current_element = ''
    for element, count in elements:
        if element.isalpha():
            element_count[element] += int(count) if count else 1
    
    return element_count


def TFA_essential_reactions(model, min_growth, constraints, deltaG0, sdeltaG0, solver = None, 
                            run_FBA_first = True, ignore_model_bounds = False, logger = None):
    if solver is None:
        solver = solver_instance(model)

    if run_FBA_first and not ignore_model_bounds:
        fba_essential_rxns = reframed.essential_reactions(model, min_growth, constraints, solver)
    else:
        fba_essential_rxns = []
    if logger:
        logger.info(f'Identified {len(fba_essential_rxns)} essential reactions using FBA')

    wt_solution = reframed.TFA(model, deltaG0, sdeltaG0=sdeltaG0, solver=solver, constraints=constraints, ignore_model_bounds = ignore_model_bounds)
    wt_growth = wt_solution.fobj
    if not wt_growth:
        if logger:
            logger.warning('No wild-type solution found')
            logger.info(f'constraints: %s', constraints)
            logger.info(f'wild type solutions: %s', wt_solution)
        return []

    tfa_essential_rxns = []

    for r_id in model.reactions:
        if r_id in fba_essential_rxns:
            continue
        constraints_ = {}
        constraints_.update(constraints)
        constraints_[r_id] = 0
        solution = reframed.TFA(model, deltaG0, sdeltaG0=sdeltaG0, solver=solver, constraints=constraints_, ignore_model_bounds = ignore_model_bounds)

        if (solution is not None
            and ((solution.status == Status.OPTIMAL and solution.fobj < min_growth * wt_growth)
                     or solution.status == Status.INFEASIBLE)):
            fba_essential_rxns.append(r_id)
    return tfa_essential_rxns + fba_essential_rxns


# From https://github.com/opencobra/cobrapy/blob/devel/src/cobra/core/formula.py
elements_and_molecular_weights = {
    "H": 1.007940,
    "He": 4.002602,
    "Li": 6.941000,
    "Be": 9.012182,
    "B": 10.811000,
    "C": 12.010700,
    "N": 14.006700,
    "O": 15.999400,
    "F": 18.998403,
    "Ne": 20.179700,
    "Na": 22.989770,
    "Mg": 24.305000,
    "Al": 26.981538,
    "Si": 28.085500,
    "P": 30.973761,
    "S": 32.065000,
    "Cl": 35.453000,
    "Ar": 39.948000,
    "K": 39.098300,
    "Ca": 40.078000,
    "Sc": 44.955910,
    "Ti": 47.867000,
    "V": 50.941500,
    "Cr": 51.996100,
    "Mn": 54.938049,
    "Fe": 55.845000,
    "Co": 58.933200,
    "Ni": 58.693400,
    "Cu": 63.546000,
    "Zn": 65.409000,
    "Ga": 69.723000,
    "Ge": 72.640000,
    "As": 74.921600,
    "Se": 78.960000,
    "Br": 79.904000,
    "Kr": 83.798000,
    "Rb": 85.467800,
    "Sr": 87.620000,
    "Y": 88.905850,
    "Zr": 91.224000,
    "Nb": 92.906380,
    "Mo": 95.940000,
    "Tc": 98.000000,
    "Ru": 101.070000,
    "Rh": 102.905500,
    "Pd": 106.420000,
    "Ag": 107.868200,
    "Cd": 112.411000,
    "In": 114.818000,
    "Sn": 118.710000,
    "Sb": 121.760000,
    "Te": 127.600000,
    "I": 126.904470,
    "Xe": 131.293000,
    "Cs": 132.905450,
    "Ba": 137.327000,
    "La": 138.905500,
    "Ce": 140.116000,
    "Pr": 140.907650,
    "Nd": 144.240000,
    "Pm": 145.000000,
    "Sm": 150.360000,
    "Eu": 151.964000,
    "Gd": 157.250000,
    "Tb": 158.925340,
    "Dy": 162.500000,
    "Ho": 164.930320,
    "Er": 167.259000,
    "Tm": 168.934210,
    "Yb": 173.040000,
    "Lu": 174.967000,
    "Hf": 178.490000,
    "Ta": 180.947900,
    "W": 183.840000,
    "Re": 186.207000,
    "Os": 190.230000,
    "Ir": 192.217000,
    "Pt": 195.078000,
    "Au": 196.966550,
    "Hg": 200.590000,
    "Tl": 204.383300,
    "Pb": 207.200000,
    "Bi": 208.980380,
    "Po": 209.000000,
    "At": 210.000000,
    "Rn": 222.000000,
    "Fr": 223.000000,
    "Ra": 226.000000,
    "Ac": 227.000000,
    "Th": 232.038100,
    "Pa": 231.035880,
    "U": 238.028910,
    "Np": 237.000000,
    "Pu": 244.000000,
    "Am": 243.000000,
    "Cm": 247.000000,
    "Bk": 247.000000,
    "Cf": 251.000000,
    "Es": 252.000000,
    "Fm": 257.000000,
    "Md": 258.000000,
    "No": 259.000000,
    "Lr": 262.000000,
    "Rf": 261.000000,
    "Db": 262.000000,
    "Sg": 266.000000,
    "Bh": 264.000000,
    "Hs": 277.000000,
    "Mt": 268.000000,
    "Ds": 281.000000,
    "Rg": 272.000000,
    "Cn": 285.000000,
    "Uuq": 289.000000,
    "Uuh": 292.000000,
}