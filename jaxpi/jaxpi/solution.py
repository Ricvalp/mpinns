import numpy as np

def get_final_solution(charts, boundaries, boundary_indices, u_preds):
    
    
    
    solutions = []
    solutions_membership = []
    
    for key in u_preds.keys():
        
        pass
    
    sol = {}
    for key in charts.keys():
        
        if key not in sol:
            sol[key] = np.zeros_like(u_preds[key])
            
            if key in boundaries:
                for boundary_key in boundaries[key].keys():
                    sol[key][boundary_indices]
                