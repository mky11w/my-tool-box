import numpy as np
from gurobipy import Model, GRB
import math

# Global best solution variables.
best_obj = -float("inf")
best_solution = None
best_constraints = None

# Define your variables, objective coefficients, and constraints globally.

VAR_NAMES = ["x", "y", "z", "m", "n", "k"]
OBJ_COEFFS = {"x": 141, "y": 187, "z": 121, "m": 83, "n": 262, "k": 127}

# Each constraint is a tuple: (coeff_dict, rhs, sense, constraint_name)
# The coeff_dict maps each variable to its coefficient (0 if not present).
CONSTRAINTS = [
    ({"x": 25, "y": 35, "z": 15, "m": 20, "n": 25, "k": 20}, 75, "<=", "c1"),
    ({"x": 20, "y": 0,  "z": 15, "m": 5,  "n": 20, "k": 30}, 50, "<=", "c2"),
    ({"x": 15, "y": 0,  "z": 15, "m": 5,  "n": 20, "k": 30}, 50, "<=", "c3"),
    ({"x": 10, "y": 30, "z": 15, "m": 5,  "n": 20, "k": 40}, 50, "<=", "c4"),
    ({"x": 1, "y": 0, "z": 0, "m": 0,  "n": 0, "k": 0}, 1, "<=", "c4"),
    ({"x": 0, "y": 1, "z": 0, "m": 0,  "n": 0, "k": 0}, 1, "<=", "c4"),
    ({"x": 0, "y": 0, "z": 1, "m": 0,  "n": 0, "k": 0}, 1, "<=", "c4"),
    ({"x": 0, "y": 0, "z": 0, "m": 1,  "n": 0, "k": 0}, 1, "<=", "c4"),
    ({"x": 0, "y": 0, "z": 0, "m": 0,  "n": 1, "k": 0}, 1, "<=", "c4"),
    ({"x": 0, "y": 0, "z": 0, "m": 0,  "n": 0, "k": 1}, 1, "<=", "c4"),
]


def create_model(vtype):
    """
    Create a Gurobi model based on global VAR_NAMES, OBJ_COEFFS, and CONSTRAINTS.
    The variable type (continuous or integer) is determined by vtype.
    """
    model = Model("ILP_model")
    model.setParam('OutputFlag', 0)
    
    # Create variables dynamically.
    variables = {}
    for name in VAR_NAMES:
        variables[name] = model.addVar(lb=0, vtype=vtype, name=name)
    model.update()
    
    # Set the objective: maximize sum(coeff * var for each var)
    obj_expr = sum(OBJ_COEFFS[name] * variables[name] for name in VAR_NAMES)
    model.setObjective(obj_expr, GRB.MAXIMIZE)
    
    # Add base constraints dynamically.
    for coeff_dict, rhs, sense, cname in CONSTRAINTS:
        constr_expr = sum(coeff_dict.get(name, 0) * variables[name] for name in VAR_NAMES)
        if sense == "<=":
            model.addConstr(constr_expr <= rhs, cname)
        elif sense == ">=":
            model.addConstr(constr_expr >= rhs, cname)
        elif sense == "=":
            model.addConstr(constr_expr == rhs, cname)
    model.update()
    return model, variables


def solve_ilp_direct():
    """
    Solve the ILP directly (using integer variables) with the shared formulation.
    Returns the optimal integer solution.
    """
    model, variables = create_model(GRB.INTEGER)
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        sol = {name: int(round(variables[name].X)) for name in VAR_NAMES}
        sol["obj"] = model.objVal
        return sol
    else:
        return None


if __name__ == "__main__":

    

    direct_solution = solve_ilp_direct()
    if direct_solution:
        print("\nDirect ILP solution (using GRB.INTEGER):")
        print("  Solution:", direct_solution)
    else:
        print("No direct ILP solution found.")
    
