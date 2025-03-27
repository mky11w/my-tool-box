from scipy.optimize import root

def equations(vars):
    x, y = vars
    eq1 = 30 * y**(-0.4) * x**(0.4) - 40 * x**(-0.6) * y**(0.6)
    # For a well-determined system, you need as many equations as unknowns.
    # Here we add a second equation to constrain the system:
    eq2 = x+2*y-200
    return [eq1, eq2]

initial_guess = [100, 75]  # Based on the expected relationship (y should be 0.75x)
solution = root(equations, initial_guess)
print("Solution (x, y):", solution.x)
