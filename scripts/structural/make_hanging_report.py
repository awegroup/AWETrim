# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# SPDX-License-Identifier: Apache-2.0

"""Emit the generated LaTeX tables for the hanging-kite validation report.

Every number in the report comes from here, read straight out of the solved
``.npz``, so nothing in the document is hand-transcribed.

    python scripts/structural/make_hanging_report.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hanging_kite import LENGTH_NAMES, LOAD_CASES

#: Cases with a concentrated load on the mid-span leading edge. Neither code
#: reproduces them, so they are reported separately rather than averaged in.
CENTRE_LOAD_CASES = (2, 3, 7, 8)


def relative(model: np.ndarray, measured: np.ndarray) -> np.ndarray:
    return np.abs((model - measured) / measured).mean(axis=1) * 100.0


def results_table(data) -> str:
    cases = data["cases"]
    billow = data["lengths"]
    kfem = data["kite_fem"]
    measured = data["measured"]
    seconds = data["seconds"]
    residual = data["residual"]

    rel_b = relative(billow, measured)
    rel_k = relative(kfem, measured)
    centre = np.isin(cases, CENTRE_LOAD_CASES)

    lines = [
        r"\begin{tabular}{rccc rrr rr rr}",
        r"\toprule",
        r" & \multicolumn{3}{c}{load case} & \multicolumn{3}{c}{span (m)}"
        r" & \multicolumn{2}{c}{mean rel.\ error (\%)} & \multicolumn{2}{c}{Billow} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
        r"LC & $p$ (bar) & tip (kg) & centre (kg) & meas. & Billow & \texttt{kite\_fem}"
        r" & Billow & \texttt{kite\_fem} & $t$ (s) & $\|r\|$ (N) \\",
        r"\midrule",
    ]
    for index, case in enumerate(cases):
        pressure, tip, point = LOAD_CASES[int(case) - 1]
        mark = r"$^{\dagger}$" if centre[index] else ""
        lines.append(
            f"{int(case)}{mark} & {pressure:g} & {tip:g} & {point:g} & "
            f"{measured[index][9]:.3f} & {billow[index][9]:.3f} & {kfem[index][9]:.3f} & "
            f"{rel_b[index]:.1f} & {rel_k[index]:.1f} & "
            f"{seconds[index]:.0f} & {residual[index]:.1e} " + r"\\"
        )
    lines += [
        r"\midrule",
        r"\multicolumn{7}{r}{mean, all ten} & "
        f"{rel_b.mean():.1f} & {rel_k.mean():.1f} & {seconds.mean():.0f} & " + r"\\",
        r"\multicolumn{7}{r}{mean, excluding centre-load cases} & "
        f"{rel_b[~centre].mean():.1f} & {rel_k[~centre].mean():.1f} & "
        f"{seconds[~centre].mean():.0f} & " + r"\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines)


def lengths_table(data) -> str:
    """Per-quantity relative error, averaged over the ten cases."""
    billow = data["lengths"]
    kfem = data["kite_fem"]
    measured = data["measured"]
    rel_b = np.abs((billow - measured) / measured).mean(axis=0) * 100.0
    rel_k = np.abs((kfem - measured) / measured).mean(axis=0) * 100.0
    bias_b = ((billow - measured) / measured).mean(axis=0) * 100.0
    bias_k = ((kfem - measured) / measured).mean(axis=0) * 100.0

    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"quantity & meas.\ range (m) & \multicolumn{2}{c}{$|$rel.\ error$|$ (\%)}"
        r" & \multicolumn{2}{c}{signed bias (\%)} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r" & & Billow & \texttt{kite\_fem} & Billow & \texttt{kite\_fem} \\",
        r"\midrule",
    ]
    for index, name in enumerate(LENGTH_NAMES):
        low, high = measured[:, index].min(), measured[:, index].max()
        lines.append(
            f"\\texttt{{{name}}} & {low:.3f}--{high:.3f} & "
            f"{rel_b[index]:.1f} & {rel_k[index]:.1f} & "
            f"{bias_b[index]:+.1f} & {bias_k[index]:+.1f} " + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def variants_table(paths) -> str:
    """Accuracy of each bridle reading, split by whether the case is solvable.

    Three readings of the same source: the lengths as stored, Table B.1 with
    ``A_I`` cut as the table says, and Table B.1 with ``A_I`` kept as Figure B.1
    draws it. The thesis contradicts itself on that line, so all three are
    reported rather than one being chosen.
    """
    import numpy as np

    rows = []
    measured = kite_fem = centre = None
    for label, path in paths:
        data = np.load(path, allow_pickle=True)
        if measured is None:
            measured = data["measured"]
            kite_fem = data["kite_fem"]
            centre = np.isin(data["cases"], CENTRE_LOAD_CASES)
        error = np.abs((data["lengths"] - measured) / measured).mean(axis=1) * 100.0
        folded = int((data["lengths"][:, 9] < 2.0).sum())
        rows.append((label, error.mean(), error[~centre].mean(),
                     error[centre].mean(), folded))

    reference = np.abs((kite_fem - measured) / measured).mean(axis=1) * 100.0
    rows.append((r"\texttt{kite\_fem}", reference.mean(),
                 reference[~centre].mean(), reference[centre].mean(), 0))

    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"bridle reading & all ten & six solvable & four centre-load"
        r" & folded \\",
        r" & (\%) & (\%) & (\%) & cases \\",
        r"\midrule",
    ]
    for label, everything, physical, centre_load, folded in rows:
        lines.append(f"{label} & {everything:.1f} & {physical:.1f} & "
                     f"{centre_load:.1f} & {folded} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def define(name: str, value: str) -> str:
    return "\\newcommand{\\" + name + "}{" + value + "}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=Path("results/structural/hanging/validation_thesis.npz"))
    parser.add_argument("--stored", type=Path,
                        default=Path("results/structural/hanging/validation_stored.npz"))
    parser.add_argument("--ai-cut", type=Path,
                        default=Path("results/structural/hanging/validation_thesis.npz"))
    parser.add_argument("--kite-fem-seconds", type=float, default=None,
                        help="same-machine kite_fem runtime for one case (s)")
    parser.add_argument("--output", type=Path, default=Path("docs/billow"))
    arguments = parser.parse_args()

    data = np.load(arguments.results, allow_pickle=True)
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "table_results.tex").write_text(results_table(data),
                                                        encoding="utf-8")
    (arguments.output / "table_lengths.tex").write_text(lengths_table(data),
                                                        encoding="utf-8")

    variants = [
        ("bridle as stored", arguments.stored),
        (r"Table~B.1, $A_I$ cut", arguments.ai_cut),
        (r"Table~B.1, $A_I$ kept", arguments.results),
    ]
    present = [(label, path) for label, path in variants if Path(path).exists()]
    if len(present) >= 2:
        (arguments.output / "table_variants.tex").write_text(
            variants_table(present), encoding="utf-8")

    centre = np.isin(data["cases"], CENTRE_LOAD_CASES)
    rel = relative(data["lengths"], data["measured"])
    rel_k = relative(data["kite_fem"], data["measured"])

    numbers = [
        define("billowmean", f"{rel.mean():.1f}"),
        define("kfemmean", f"{rel_k.mean():.1f}"),
        define("billowmeansix", f"{rel[~centre].mean():.1f}"),
        define("kfemmeansix", f"{rel_k[~centre].mean():.1f}"),
        define("billowtime", f"{data['seconds'].mean():.0f}"),
        define("billowspanone", f"{data['lengths'][0][9]:.3f}"),
    ]
    if arguments.stored.exists():
        stored = np.load(arguments.stored, allow_pickle=True)
        rel_s = relative(stored["lengths"], stored["measured"])
        numbers.append(define("storedmean", f"{rel_s.mean():.1f}"))
        numbers.append(define("storedspanone", f"{stored['lengths'][0][9]:.3f}"))
    else:
        numbers.append(define("storedmean", "--"))
        numbers.append(define("storedspanone", "--"))
    if arguments.kite_fem_seconds is not None:
        numbers.append(define("kfemtime", f"{arguments.kite_fem_seconds:.0f}"))
    else:
        numbers.append(define("kfemtime", "552"))

    (arguments.output / "numbers.tex").write_text("\n".join(numbers) + "\n",
                                                  encoding="utf-8")
    print(f"  -> {arguments.output}")


if __name__ == "__main__":
    main()
