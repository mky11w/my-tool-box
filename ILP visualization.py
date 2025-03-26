import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from gurobipy import Model, GRB
import math

# Global variables to record the best integer solution found in branch-and-bound.
best_obj = -float("inf")
best_solution = None
best_constraints = None

def solve_lp(additional_constraints):
    """
    Create and solve the LP relaxation for:
      maximize x + y
      subject to: x + 2y <= 4,  3x + y <= 5,  x, y >= 0,
    with extra branching constraints provided in additional_constraints.
    
    additional_constraints is a list of tuples of the form:
       (var_name, bound, sense)
    where sense is "<=" or ">=".
    """
    model = Model("LP_relaxation")
    model.setParam('OutputFlag', 0)
    
    # Define continuous variables.
    x = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x")
    y = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="y")
    model.update()
    
    # Base constraints.
    model.addConstr(x + 2*y <= 4, "c1")
    model.addConstr(3*x + y <= 5, "c2")
    
    # Add extra (branching) constraints.
    for cons in additional_constraints:
        var_name, bound, sense = cons
        var = x if var_name == 'x' else y
        if sense == "<=":
            model.addConstr(var <= bound)
        elif sense == ">=":
            model.addConstr(var >= bound)
    
    # Set the objective.
    model.setObjective(x + y, GRB.MAXIMIZE)
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        return {"x": x.X, "y": y.X, "obj": model.objVal}
    else:
        return None

def branch_and_bound(additional_constraints, node_name, tree, depth=0, verbose=True):
    """
    Recursively solve the LP relaxation with the current additional_constraints.
    If the LP solution is fractional, pick a fractional variable and branch.
    The tree dict will record nodes (with labels and children) for visualization.
    The verbose flag controls whether intermediate messages are printed.
    """
    global best_obj, best_solution, best_constraints
    sol = solve_lp(additional_constraints)
    indent = "  " * depth  # For console printing.
    
    if sol is None:
        label = f"{node_name}\nInfeasible"
        tree[node_name] = {"label": label, "children": []}
        if verbose:
            print(indent + f"{node_name}: Infeasible")
        return None

    cons_str = ", ".join([f"{c[0]} {c[2]} {c[1]}" for c in additional_constraints]) if additional_constraints else "None"
    label = (f"{node_name}\nConstraints: {cons_str}\n"
             f"Sol: (x={sol['x']:.2f}, y={sol['y']:.2f})\nObj: {sol['obj']:.2f}")
    
    # Check if both variables are integer.
    is_integer = (abs(sol['x'] - round(sol['x'])) < 1e-5) and (abs(sol['y'] - round(sol['y'])) < 1e-5)
    if is_integer:
        label += "\nInteger"
        tree[node_name] = {"label": label, "children": []}
        if verbose:
            print(indent + f"{node_name}: Integer solution found with obj {sol['obj']:.2f}")
        if sol["obj"] > best_obj:
            best_obj = sol["obj"]
            best_solution = sol
            best_constraints = additional_constraints.copy()
        return sol
    else:
        tree[node_name] = {"label": label, "children": []}
        if verbose:
            print(indent + f"{node_name}: Fractional solution (x={sol['x']:.2f}, y={sol['y']:.2f}) with obj {sol['obj']:.2f}")
        # Choose a variable to branch on – branch on x if fractional, otherwise y.
        if abs(sol['x'] - round(sol['x'])) >= 1e-5:
            branch_var = 'x'
            value = sol['x']
        else:
            branch_var = 'y'
            value = sol['y']
        
        # Left branch: variable <= floor(value)
        left_constraints = additional_constraints.copy()
        left_constraints.append((branch_var, int(np.floor(value)), "<="))
        left_name = node_name + "L"
        branch_and_bound(left_constraints, left_name, tree, depth+1, verbose)
        tree[node_name]["children"].append(left_name)
        
        # Right branch: variable >= ceil(value)
        right_constraints = additional_constraints.copy()
        right_constraints.append((branch_var, int(np.ceil(value)), ">="))
        right_name = node_name + "R"
        branch_and_bound(right_constraints, right_name, tree, depth+1, verbose)
        tree[node_name]["children"].append(right_name)
        
        return None

def visualize_tree(tree):
    """
    Use networkx to visualize the branch-and-bound tree.
    Each node shows its name, the extra constraints, the LP solution, and the objective value.
    """
    G = nx.DiGraph()
    for node, data in tree.items():
        G.add_node(node, label=data["label"])
        for child in data["children"]:
            G.add_edge(node, child)
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, 'label')
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=3000,
            node_color='lightblue', font_size=8, arrows=True)
    plt.title("Branch and Bound Tree")
    plt.show()

def solve_ilp_gurobi():
    """
    Solve the ILP directly with Gurobi as a MIP:
      maximize x + y
      subject to: x + 2y <= 4,  3*x + y <= 5,  x, y integer and >= 0.
    Returns the optimal solution.
    """
    model = Model("ILP")
    model.setParam('OutputFlag', 0)
    
    # Define integer variables.
    x = model.addVar(lb=0, vtype=GRB.INTEGER, name="x")
    y = model.addVar(lb=0, vtype=GRB.INTEGER, name="y")
    model.update()
    
    # Add constraints.
    model.addConstr(x + 2*y <= 4, "c1")
    model.addConstr(3*x + y <= 5, "c2")
    
    # Set the objective.
    model.setObjective(x + y, GRB.MAXIMIZE)
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        return {"x": x.X, "y": y.X, "obj": model.objVal}
    else:
        return None

if __name__ == "__main__":
    # Run branch-and-bound with verbose output.
    tree = {}
    initial_constraints = []  # No branching constraints at the root.
    branch_and_bound(initial_constraints, "root", tree, verbose=True)
    
    # Solve the ILP directly with Gurobi.
    direct_solution = solve_ilp_gurobi()
    
    # Use math.isclose for tolerance-based comparison.
    valid = False
    if best_solution and direct_solution:
        if (math.isclose(best_solution['x'], direct_solution['x'], rel_tol=1e-5) and 
            math.isclose(best_solution['y'], direct_solution['y'], rel_tol=1e-5)):
            valid = True

    if valid:
        # If validation is successful, display both solutions and the tree.
        print("\nValidation successful: Branch-and-bound solution matches the direct Gurobi ILP solution.")
        print("Best branch-and-bound solution:")
        print(f"  x = {best_solution['x']:.2f}, y = {best_solution['y']:.2f}, Obj = {best_obj:.2f}")
        print("\nDirect Gurobi ILP solution:")
        print(f"  x = {direct_solution['x']:.2f}, y = {direct_solution['y']:.2f}, Obj = {direct_solution['obj']:.2f}")
        visualize_tree(tree)
    else:
        # If validation fails, do not draw the tree or display branch-and-bound details.
        print("\nValidation failed: This is the direct Gurobi ILP solution:")
        if direct_solution:
            print(f"  x = {direct_solution['x']:.2f}, y = {direct_solution['y']:.2f}, Obj = {direct_solution['obj']:.2f}")
        else:
            print("No direct solution found.")
