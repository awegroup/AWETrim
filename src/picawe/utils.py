# import casadi as ca
# def substitute_knowns(func, known_state):
#     """
#     Substitute known values into a CasADi function and keep symbolic variables already defined.

#     :param func: Original CasADi function (e.g., residual_func).
#     :param known_state: Dictionary of known parameter values to substitute.
#     :return: A new CasADi function with reduced inputs.
#     """
#     # Retrieve symbolic inputs and their names
#     symbolic_inputs_all = func.sx_in()
#     symbolic_names_all = func.name_in()

#     # Substitute known parameters in the function output
#     residual_expr = func.call(symbolic_inputs_all)[0]
#     for name, value in known_state.items():
#         if name in symbolic_names_all:
#             idx = symbolic_names_all.index(name)
#             residual_expr = ca.substitute(residual_expr, symbolic_inputs_all[idx], value)
#             print(f"Substituting known value '{name}' = {value}")

#     # Filter remaining symbolic inputs
#     remaining_inputs = []
#     remaining_names = []
#     for i, name in enumerate(symbolic_names_all):
#         if name not in known_state:
#             remaining_inputs.append(symbolic_inputs_all[i])
#             remaining_names.append(name)

#     # Return the reduced function
#     return ca.Function(func.name() + '_reduced', remaining_inputs, [residual_expr], remaining_names, func.name_out())
