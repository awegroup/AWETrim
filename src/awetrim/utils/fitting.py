# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# Licensed under the EUPL, Version 1.2 or - as soon they will be approved by
# the European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
#     https://joinup.ec.europa.eu/software/page/eupl
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the Licence is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the Licence for the specific language governing permissions and
# limitations under the Licence.
#
# SPDX-License-Identifier: EUPL-1.2

import numpy as np


def compute_weighted_least_squares(y, A, W=None):
    if W is None:
        x_hat = np.linalg.inv(A.T @ A) @ A.T @ y
    else:
        x_hat = np.linalg.inv(A.T @ W @ A) @ A.T @ W @ y
    return x_hat


def construct_A_matrix(dependencies, **kwargs):
    """
    Constructs the A matrix based on the dependencies provided.

    Parameters:
        dependencies (list): List of strings representing dependencies for the model.
                             Each string should be a valid Python expression involving the inputs.
        kwargs (dict): Dictionary of inputs where keys match variable names in dependencies.

    Returns:
        np.array: The A matrix for the regression.
    """
    A = []
    global_scope = {"np": np}  # Include np in global scope for eval
    global_scope.update(kwargs)  # Add input variables to the scope

    for dep in dependencies:
        term = eval(dep, global_scope)
        A.append(term)

    return np.vstack(A).T


def fit_and_evaluate_model(data, dependencies, **kwargs):
    """
    Fits a model using weighted least squares and prints mean squared error.

    Parameters:
        data (np.array): The dependent variable data (e.g., CL, CD).
        dependencies (list): List of dependencies in string format for model construction.
        weights (np.array): Weight matrix for the weighted least squares calculation.
        kwargs (dict): Dictionary of inputs like alpha, up, us, etc. required by the dependencies.

    Returns:
        dict: Coefficients and Mean Squared Error (MSE).
    """
    # Construct A matrix with the specified dependencies
    A = construct_A_matrix(dependencies, **kwargs)
    # Calculate coefficients using weighted least squares
    coeffs = compute_weighted_least_squares(data, A)
    # Calculate estimated values
    data_est = A @ coeffs
    # Mean Squared Error
    mse = np.mean((data - data_est) ** 2)
    # Print results
    # print(f"Coefficients: {coeffs}")
    # print(f"Mean Squared Error: {mse}")
    return {"coeffs": coeffs, "MSE": mse, "data_est": data_est}
