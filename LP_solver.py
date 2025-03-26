from gurobipy import Model, GRB

def gurobi(c, A, b, signs):
    """
    Validates the solution obtained from the simplex method using the Gurobi optimizer.
    
    Parameters:
      c     : List of coefficients in the objective function.
      A     : Coefficient matrix for the constraints.
      b     : Right-hand side values for the constraints.
      signs : List of constraint types ('<=', '>=', or '=').
    """
    model = Model("Validation")
    # Optionally suppress Gurobi output.
    model.Params.OutputFlag = 0
    
    # Add decision variables
    x = [model.addVar(lb=0, name=f"x{i+1}") for i in range(len(c))]
    
    # Set objective function
    model.setObjective(sum(c[i] * x[i] for i in range(len(c))), GRB.MAXIMIZE)
    
    # Add constraints based on their signs
    for i in range(len(A)):
        expr = sum(A[i][j] * x[j] for j in range(len(c)))
        
        if signs[i] == '<=':
            model.addConstr(expr <= b[i], f"constraint_{i}")
        elif signs[i] == '>=':
            model.addConstr(expr >= b[i], f"constraint_{i}")
        elif signs[i] == '=':
            model.addConstr(expr == b[i], f"constraint_{i}")
        else:
            raise ValueError(f"Unsupported constraint sign: {signs[i]}")
    
    # Optimize the model
    model.optimize()
    
    print("\n[Gurobi] Validation:")
    if model.status == GRB.OPTIMAL:
        for var in x:
            print(f"{var.varName} = {var.x:.6f}")
        print(f"Objective Value = {model.objVal:.6f}")
    else:
        print("No optimal solution found by Gurobi.")

if __name__ == "__main__":
    # Test input with mixed constraint types
    c = [4, 3, 3]  # objective function coefficients
    A = [
        [4, 2, 1],
        [3, 4, 2],
        [2, 1, 3],

    ]  # constraint matrix
    b = [10, 14, 7]
    signs = ['<=', '<=', '<=', '>=',"<="]
    
    gurobi(c, A, b, signs)