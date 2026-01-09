import reframed
import pandas as pd 
from collections import defaultdict
import numpy as np

def init_solver(solver):
    solver.problem.params.IntFeasTol = 1e-9
    solver.problem.params.IntegralityFocus = 1
    solver.problem.params.MIPFocus = 2
    solver.problem.params.FeasibilityTol = 1e-9
    solver.problem.params.OptimalityTol = 1e-9
    solver.problem.params.TimeLimit = 120

def test_model(model, species_growth_data, cs_name_to_id, method = 'FBA', 
               deltaG0 = None, sdeltaG0 = None, min_growth = 1e-2, uptake_rate = -10,
               additional_constraints = None, min_concentration = 1e-6, max_concentration = 1e-1):
    in_silico_results = {}
    pred_matrix = defaultdict(int)
    data = []
    solver = reframed.solver_instance(model)
    init_solver(solver)

    existing_cs_constraints = {}
    for cs_id in cs_name_to_id.values():
        ex_id = f'R_EX_{cs_id}_e'
        if model.reactions[ex_id].lb <= -1:
            print(f'Change lower bound of {ex_id} from {model.reactions[ex_id].lb} to 0')
            existing_cs_constraints[ex_id] = 0
            # model.set_flux_bounds(ex_id, lb=0)


    for cs, in_vitro_result in species_growth_data.items():
        cs_id = cs_name_to_id[cs]
        if np.isnan(in_vitro_result):
            # Skip this as there is noe data
            continue
        # ex_reaction = model.reactions[f'R_EX_{cs_id}_e']
        # ex_reaction.lb = -10

        constraints={f'R_EX_{cs_id}_e':uptake_rate}
        if additional_constraints:
            constraints.update({key: value for key, value in additional_constraints.items() if not constraints.get(key)})
        if len(existing_cs_constraints):
            constraints.update({key: value for key, value in existing_cs_constraints.items() if not constraints.get(key)})
            
        if method == 'FBA':
            solution = reframed.FBA(model, constraints=constraints, solver = solver)
            # print(cs_id, solution)
        elif method == 'TFA':
            solution = reframed.TFA(model, deltaG0, sdeltaG0=sdeltaG0, 
                                    ignore_model_bounds=False, constraints=constraints, solver = solver,
                                    concentration_min = min_concentration, concentration_max = max_concentration)
        
        else:
            raise NotImplementedError

        if solution.fobj:
            in_silico_results[cs] = solution.fobj
        else:
            in_silico_results[cs] = 0
        
        if in_vitro_result == 1:
            if in_silico_results[cs] > min_growth:
                pred_outcome = 'TP'
            else:
                pred_outcome = 'FN'
        elif in_vitro_result == 0:
            if in_silico_results[cs] > min_growth:
                pred_outcome = 'FP'
            else:
                pred_outcome = 'TN'
        else:
            continue
        pred_matrix[pred_outcome] += 1
        data.append([cs, cs_id, in_vitro_result, in_silico_results[cs], pred_outcome])   
    
    df = pd.DataFrame(data, columns = ['Carbon source', 'Carbon source ID',
                                       'In vitro', 'In silico', 'Pred outcome'])
    # pred_matrix = df['Pred outcome'].value_counts()
    return df, pred_matrix