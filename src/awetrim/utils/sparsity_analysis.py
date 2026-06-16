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
import casadi as ca


def stiffness_report(opti, sol=None, name="NLP"):
    # Objects
    x = opti.x
    g = opti.g
    f = opti.f
    lam = opti.lam_g

    # If no solution passed, evaluate at initial guess
    if sol is None:
        x0 = opti.initial().value(x)
        lam0 = np.zeros(opti.ng)
    else:
        x0 = sol.value(x)
        # lam_g is available only after a solve
        try:
            lam0 = sol.value(lam)
        except RuntimeError:
            lam0 = np.zeros(opti.ng)

    # Build functions
    Jg = ca.jacobian(g, x)
    Hf = ca.hessian(f, x)[0]
    # Hessian of the Lagrangian (H_f + sum_i lam_i * H_gi); cheap if many linear/affine g
    Lag = f + ca.dot(lam, g)
    HLag = ca.hessian(Lag, x)[0]

    fJg = ca.Function("fJg", [x], [Jg])
    fHf = ca.Function("fHf", [x], [Hf])
    fHLag = ca.Function("fHLag", [x, lam], [HLag])

    Jg0 = np.array(fJg(x0))
    Hf0 = np.array(fHf(x0))
    HLag0 = np.array(fHLag(x0, lam0))

    # KKT (symmetric indefinite)
    # [ HLag  Jg.T ]
    # [  Jg    0  ]
    KKT = np.block([[HLag0, Jg0.T], [Jg0, np.zeros((Jg0.shape[0], Jg0.shape[0]))]])

    # Safe condition numbers (fall back to SVD if needed)
    def cond(A):
        try:
            return np.linalg.cond(A)
        except np.linalg.LinAlgError:
            s = np.linalg.svd(A, compute_uv=False)
            return s[0] / s[-1] if s[-1] > 0 else np.inf

    print(f"=== {name} stiffness report ===")
    print("cond(Jg)          :", cond(Jg0))
    print("cond(Jg Jg^T)     :", cond(Jg0 @ Jg0.T))
    print("cond(H_f)         :", cond(Hf0))
    print("cond(H_Lagrangian):", cond(HLag0))
    print("cond(KKT)         :", cond(KKT))

    # Eigenvalue spread (magnitude range)
    def eig_spread(A):
        w = np.linalg.eigvals(A)
        aw = np.sort(np.abs(w))
        aw = aw[aw > 0]  # drop exact zeros
        if aw.size == 0:
            return (0.0, 0.0, np.inf)
        return (aw[0], aw[-1], aw[-1] / aw[0])

    emin, emax, ratio = eig_spread(HLag0)
    print("HLag |eig|min,max,ratio:", emin, emax, ratio)

    # Simple magnitude/scale hints
    print("||x||_inf:", np.max(np.abs(x0)))
    if sol is not None:
        g0 = sol.value(g)
    else:
        g0 = ca.Function("fg", [x], [g])(x0).full().squeeze()
    print("||g||_inf:", np.max(np.abs(g0)))
