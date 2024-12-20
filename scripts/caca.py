import casadi as ca

# Define the rootfinding problem
x = ca.SX.sym('x')  # Variable for the rootfinder
root_func = ca.Function('root_func', [x], [x**2 - 2])  # Solve x^2 - 2 = 0
rootfinder = ca.rootfinder('rootfinder', 'newton', root_func)

# Create an Opti problem
opti = ca.Opti()

# Define a variable in the Opti problem
y = opti.variable()

# Call the rootfinder as part of the optimization problem
z = rootfinder(y)  # z is the solution of the rootfinding problem for y

# Add an objective or constraint
opti.minimize((z - 1)**2)  # Minimize the difference of the root from 1
opti.subject_to(y > 0)     # y must be positive
opti.set_initial(y, 1.0)  # Set a reasonable initial guess

# Solve the optimization problem
opti.solver('ipopt')
solution = opti.solve()

# Extract the results
y_sol = solution.value(y)
z_sol = rootfinder(y_sol)  # Solve the rootfinding problem for the optimal y
print("Optimal y:", y_sol)
print("Rootfinder result:", z_sol)
