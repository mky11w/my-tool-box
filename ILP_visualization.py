import numpy as np
from gurobipy import Model, GRB
import math

# Global best solution variables.
best_obj = -float("inf")
best_solution = None
best_constraints = None

# Define your variables, objective coefficients, and constraints globally.
# Change these lists/dicts as needed for different problems.
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

def display_tableau(model):
    """
    Display a summary of the current LP model's constraints.
    """
    print("\n--- Tableau ---")
    for constr in model.getConstrs():
        row = model.getRow(constr)
        terms = []
        for j in range(row.size()):
            coeff = row.getCoeff(j)
            varname = row.getVar(j).VarName
            terms.append(f"{coeff:.2f}*{varname}")
        expr_str = " + ".join(terms)
        slack = constr.getAttr("Slack")
        dual  = constr.getAttr("Pi")
        print(f"{constr.ConstrName}: {expr_str} {constr.Sense} {constr.RHS:.2f} "
              f"(Slack: {slack:.2f}, Dual: {dual:.2f})")
    print("---------------\n")

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

def solve_lp(additional_constraints):
    """
    Solve the LP relaxation (continuous variables) with extra branching constraints.
    additional_constraints is a list of tuples: (var_name, bound, sense)
    """
    model, variables = create_model(GRB.CONTINUOUS)
    
    # Add extra branching constraints.
    for var_name, bound, sense in additional_constraints:
        if var_name in variables:
            if sense == "<=":
                model.addConstr(variables[var_name] <= bound)
            elif sense == ">=":
                model.addConstr(variables[var_name] >= bound)
    model.update()
    
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        display_tableau(model)
        # Build solution for all variables.
        sol = {name: variables[name].X for name in VAR_NAMES}
        sol["obj"] = model.objVal
        return sol
    else:
        return None

def branch_and_bound(additional_constraints, depth=0, verbose=True):
    """
    Recursively solve the LP relaxation with given extra constraints.
    If a variable has a fractional value, branch on it.
    Records the best integer solution found.
    """
    global best_obj, best_solution, best_constraints
    sol = solve_lp(additional_constraints)
    indent = "  " * depth  # For neat printing.
    
    if sol is None:
        if verbose:
            print(indent + f"Infeasible node with constraints: {additional_constraints}")
        return None

    if verbose:
        print(indent + f"Node with constraints {additional_constraints}: sol = {sol}")
        
    # Prune if the LP objective is no better than the best known integer solution.
    if sol["obj"] <= best_obj:
        if verbose:
            print(indent + f"Pruned node (LP obj {sol['obj']:.2f} <= best obj {best_obj:.2f}).")
        return None
    
    # Check for any fractional variables.
    fractional_var = None
    for name in VAR_NAMES:
        if abs(sol[name] - round(sol[name])) > 1e-5:
            fractional_var = name
            value = sol[name]
            break
    
    if fractional_var is None:
        # An integer solution is found.
        if verbose:
            print(indent + f"Integer solution found: {sol}")
        if sol["obj"] > best_obj:
            best_obj = sol["obj"]
            best_solution = sol
            best_constraints = additional_constraints.copy()
        return sol
    else:
        # Branch on the fractional variable.
        left_constraints = additional_constraints.copy()
        left_bound = int(math.floor(value))
        left_constraints.append((fractional_var, left_bound, "<="))
        if verbose:
            print(indent + f"Branching on {fractional_var} = {value:.2f}: "
                  f"left branch with {fractional_var} <= {left_bound}")
        branch_and_bound(left_constraints, depth+1, verbose)
        
        right_constraints = additional_constraints.copy()
        right_bound = int(math.ceil(value))
        right_constraints.append((fractional_var, right_bound, ">="))
        if verbose:
            print(indent + f"Branching on {fractional_var} = {value:.2f}: "
                  f"right branch with {fractional_var} >= {right_bound}")
        branch_and_bound(right_constraints, depth+1, verbose)
        
    return None

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

def check_gurobi_setup():
    """
    Check that Gurobi is set up correctly by solving a trivial LP.
    """
    try:
        test_model, variables = create_model(GRB.CONTINUOUS)
        # For example, maximize the first variable.
        test_model.setObjective(variables[VAR_NAMES[0]], GRB.MAXIMIZE)
        test_model.optimize()
        if test_model.status == GRB.OPTIMAL:
            print("Gurobi is set up correctly.\n")
        else:
            print("Gurobi setup issue: Test model not optimal.\n")
    except Exception as e:
        print("Gurobi setup error:", e)

if __name__ == "__main__":
    # Verify Gurobi is correctly installed.
    check_gurobi_setup()
    
    # Run branch-and-bound using LP relaxations.
    print("Starting branch-and-bound procedure...\n")
    branch_and_bound(additional_constraints=[])
    
    if best_solution is not None:
        print("\nBranch-and-bound best integer solution:")
        print("  Solution:", best_solution)
        print("  Objective value:", best_obj)
        print("Extra constraints applied at this node:", best_constraints)
    else:
        print("\nNo integer solution found via branch-and-bound.")
    
    # Solve the ILP directly (using GRB.INTEGER).
    direct_solution = solve_ilp_direct()
    if direct_solution:
        print("\nDirect ILP solution (using GRB.INTEGER):")
        print("  Solution:", direct_solution)
    else:
        print("No direct ILP solution found.")
    
    # Validate that both methods match.
    if best_solution and direct_solution:
        is_match = all(best_solution[name] == direct_solution[name] for name in VAR_NAMES)
        if is_match:
            print("\nValidation successful: Branch-and-bound and direct ILP solutions match.")
        else:
            print("\nValidation warning: Solutions differ between branch-and-bound and direct ILP.")
