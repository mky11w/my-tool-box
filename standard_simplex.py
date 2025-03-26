import numpy as np
import time
from gurobipy import Model, GRB

# Set printing options: fixed point with 6 decimals and no scientific notation.
np.set_printoptions(precision=2, suppress=True, floatmode='fixed')

def print_tableau(tableau, basic_vars, non_basic_vars):
    """
    Prints the current simplex tableau with headers.
    """
    print("\nCurrent Tableau:")
    header = non_basic_vars + basic_vars + ['RHS']
    print("\t".join(header))
    for row in tableau:
        print("\t".join(f"{val:.2f}" for val in row))

def find_pivot_column(tableau):
    """
    Identifies the pivot column by finding the most negative coefficient in the objective row.
    Returns None if no negative coefficients are found (optimal solution reached).
    """
    last_row = tableau[-1, :-1]
    min_val = np.min(last_row)
    if min_val >= 0:
        return None  # Optimal solution reached.
    return np.argmin(last_row)

def find_pivot_row(tableau, pivot_col):
    """
    Determines the pivot row using the minimum ratio test.
    Returns None if the problem is unbounded.
    """
    ratios = []
    for i in range(len(tableau) - 1):  # Exclude the objective row.
        col_val = tableau[i, pivot_col]
        if col_val <= 0:
            ratios.append(np.inf)
        else:
            ratios.append(tableau[i, -1] / col_val)
    min_ratio = np.min(ratios)
    if np.isinf(min_ratio):
        return None  # The linear program is unbounded.
    return ratios.index(min_ratio)

def pivot(tableau, pivot_row, pivot_col):
    """
    Performs the pivot operation on the tableau.
    """
    tableau[pivot_row] = tableau[pivot_row] / tableau[pivot_row, pivot_col]
    for i in range(len(tableau)):
        if i != pivot_row:
            tableau[i] = tableau[i] - tableau[i, pivot_col] * tableau[pivot_row]
    return tableau

def simplex(c, A, b, signs, delay=0.5):
    """
    Solves a linear programming problem using the standard simplex method (for '≤' constraints only).
    
    Parameters:
      c      : List of coefficients in the objective function.
      A      : Coefficient matrix for the constraints.
      b      : Right-hand side values for the constraints.
      signs  : List of constraint signs (only '<=' supported).
      delay  : Time delay (in seconds) between iterations for visualization.
    
    Returns:
      A dictionary containing the solution values for all variables.
    """
    num_vars = len(c)
    num_constraints = len(b)
    
    tableau_rows = []
    
    # Construct the initial tableau rows.
    # For each constraint, append the slack variable(s) and the RHS.
    for i, sign in enumerate(signs):
        if sign != '<=':
            raise NotImplementedError("Standard simplex currently supports only '<=' constraints.")
        row = list(A[i])  # Coefficients for decision variables.
        slack = [0] * num_constraints
        slack[i] = 1      # Add the slack variable.
        row += slack
        row += [b[i]]     # Append RHS.
        tableau_rows.append(row)
        
    num_slack = num_constraints

    # Build the initial tableau as a NumPy array.
    tableau = np.array(tableau_rows, dtype=float)
    # Objective row: negative coefficients (for maximization) and zeros for slack variables.
    obj_row = [-val for val in c] + [0]*num_slack + [0]
    tableau = np.vstack([tableau, obj_row])
    
    # Define initial variables:
    # Decision variables (x's) are non-basic; slack variables (s's) are basic.
    non_basic_vars = [f"x{i+1}" for i in range(num_vars)]
    basic_vars = [f"s{i+1}" for i in range(num_constraints)]
    
    iteration = 0
    while True:
        print(f"\n=== Iteration {iteration} ===")
        # For printing, we combine non-basic and basic vars to match the tableau columns.
        print_tableau(tableau, basic_vars, non_basic_vars)
        time.sleep(delay)
        
        pivot_col = find_pivot_column(tableau)
        if pivot_col is None:
            print("\nOptimal solution found.")
            break

        pivot_row = find_pivot_row(tableau, pivot_col)
        if pivot_row is None:
            print("\nThe linear program is unbounded.")
            return None

        # For clarity, determine the entering variable.
        # Since our tableau columns are [x's | s's | RHS],
        # if pivot_col is less than the number of decision variables, it refers to an x variable,
        # otherwise it refers to a slack variable.
        if pivot_col < len(non_basic_vars):
            pivot_var = non_basic_vars[pivot_col]
        else:
            pivot_var = basic_vars[pivot_col - len(non_basic_vars)]
        leaving_var = basic_vars[pivot_row]
        print(f"Pivot Element at Row {pivot_row}, Column {pivot_col}")
        print(f"Entering Variable: {pivot_var}, Leaving Variable: {leaving_var}")

        # Update the basic variables and perform the pivot operation.
        basic_vars[pivot_row] = pivot_var
        tableau = pivot(tableau, pivot_row, pivot_col)
        iteration += 1

    print("\nFinal Tableau:")
    print_tableau(tableau, basic_vars, non_basic_vars)

    # Extract solution: decision variables (x's) and slack variables.
    solution = {var: 0 for var in non_basic_vars + basic_vars}
    for i, var in enumerate(basic_vars):
        solution[var] = tableau[i, -1]
    
    print("\nSimplex Solution (Standard Simplex):")
    for var in sorted(solution.keys()):
        if var.startswith("x"):
            print(f"{var} = {solution[var]:.6f}")
    print(f"Objective Value = {tableau[-1, -1]:.6f}")

    return solution

def validate_with_gurobi(c, A, b, signs):
    """
    Validates the solution obtained from the simplex method using the Gurobi optimizer.
    
    Parameters:
      c     : List of coefficients in the objective function.
      A     : Coefficient matrix for the constraints.
      b     : Right-hand side values for the constraints.
      signs : List of constraint signs (only '<=' supported).
    """
    model = Model("Validation")
    # Optionally suppress Gurobi output.
    model.Params.OutputFlag = 0

    x = [model.addVar(lb=0, name=f"x{i+1}") for i in range(len(c))]
    model.setObjective(sum(c[i] * x[i] for i in range(len(c))), GRB.MAXIMIZE)

    for i in range(len(A)):
        expr = sum(A[i][j] * x[j] for j in range(len(c)))
        if signs[i] == '<=':
            model.addConstr(expr <= b[i], name=f"constr_{i+1}")
        else:
            raise NotImplementedError("Only '<=' constraints are supported for validation.")
            
    model.optimize()

    print("\n[Gurobi] Validation:")
    if model.status == GRB.OPTIMAL:
        for var in x:
            print(f"{var.varName} = {var.x:.6f}")
        print(f"Objective Value = {model.objVal:.6f}")
    else:
        print("No optimal solution found by Gurobi.")

if __name__ == "__main__":
    # Modified test input with only '<=' constraints for standard simplex.
    c = [4, 3,3] # obj func coeff
    A = [
        [4, 2,1],
        [3, 4,2],
        [2,1,3]
    ] # constraint matrix
    b = [10, 14,7]
    signs = ['<=','<=','<=']

    solution = simplex(c, A, b, signs)
    if solution is not None:
        validate_with_gurobi(c, A, b, signs)
