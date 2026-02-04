# EVOLVE-BLOCK-START
import os
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---- Connectome loader for SI5 Excel layout ----
# The SI5 sheets store labels starting at row=3/col=4 (1-indexed).
# In 0-indexed pandas header=None:
#   - column labels are at raw.iloc[2, 3:]
#   - row labels are at raw.iloc[3:, 2]
#   - numeric matrix is at raw.iloc[3:, 3:]
#
# We'll:
#   1) parse raw sheet
#   2) determine valid label extents
#   3) build dense numeric matrix
#   4) extract neuron-neuron submatrix using intersection of row/col labels


def _clean_label(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    return s if s else None


def _truncate_to_last_nonnull(labels):
    last = -1
    for i, x in enumerate(labels):
        if x is not None:
            last = i
    return labels[: last + 1]


def _load_si5_sheet_matrix(xlsx_path: str, sheet_name: str):
    raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)

    col_labels = [_clean_label(x) for x in raw.iloc[2, 3:].tolist()]
    row_labels = [_clean_label(x) for x in raw.iloc[3:, 2].tolist()]
    col_labels = _truncate_to_last_nonnull(col_labels)
    row_labels = _truncate_to_last_nonnull(row_labels)

    mat = raw.iloc[3 : 3 + len(row_labels), 3 : 3 + len(col_labels)]
    mat = mat.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)

    return row_labels, col_labels, mat


def load_connectome_neuron_submatrices():
    # Prefer an explicit env var (set by evaluator), then probe common locations.
    candidates = []
    env_path = os.environ.get("CE_CONNECTOME_PATH", "").strip()
    if env_path:
        candidates.append(env_path)

    # Common local names / layouts
    candidates += [
        "connectome.xlsx",
        "input/connectome.xlsx",
        "SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",
        "/mnt/data/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",
        # Your repo-local absolute path (safe fallback if you keep it here)
        "/home/rag/Projects/science-codeevolve/problems/elegans/input/connectome.xlsx",
    ]

    # Try to locate the repo root from the current working directory
    try:
        cwd = Path.cwd().resolve()
        for parent in [cwd] + list(cwd.parents):
            candidates.append(str(parent / "problems" / "elegans" / "input" / "connectome.xlsx"))
    except Exception:
        pass

    xlsx_path = None

    for c in candidates:
        if Path(c).exists():
            xlsx_path = str(Path(c))
            break
    if xlsx_path is None:
        raise FileNotFoundError(
            "Could not find connectome workbook. Put it next to solution.py as "
            "'connectome.xlsx' or 'SI 5 Connectome adjacency matrices, corrected July 2020.xlsx'."
        )

    # Light caching to speed repeated runs during evolution
    cache_path = Path("connectome_cache_si5_herm_nn.npz")
    try:
        mtime = Path(xlsx_path).stat().st_mtime
    except Exception:
        mtime = None

    if cache_path.exists():
        try:
            data = np.load(str(cache_path), allow_pickle=True)
            if ("mtime" in data) and (mtime is None or float(data["mtime"]) == float(mtime)):
                neurons = data["neurons"].tolist()
                chem_nn = data["chem_nn"]
                gap_nn = data["gap_nn"]
                return xlsx_path, neurons, chem_nn, gap_nn
        except Exception:
            pass  # fall through to rebuild cache

    # Load sheets
    r_c, c_c, mat_c = _load_si5_sheet_matrix(xlsx_path, "hermaphrodite chemical")
    r_g, c_g, mat_g = _load_si5_sheet_matrix(xlsx_path, "hermaphrodite gap jn symmetric")

    # Neurons: intersection of chemical rows & chemical cols, preserving row order.
    neuron_set = set(r_c) & set(c_c)
    neurons = [n for n in r_c if n in neuron_set]

    # Build indices for subsetting
    row_index_c = {n: i for i, n in enumerate(r_c)}
    col_index_c = {n: i for i, n in enumerate(c_c)}
    row_index_g = {n: i for i, n in enumerate(r_g)}
    col_index_g = {n: i for i, n in enumerate(c_g)}

    idx_r_c = [row_index_c[n] for n in neurons]
    idx_c_c = [col_index_c[n] for n in neurons]
    idx_r_g = [row_index_g[n] for n in neurons]
    idx_c_g = [col_index_g[n] for n in neurons]

    chem_nn = mat_c[np.ix_(idx_r_c, idx_c_c)]   # directed: pre (rows) -> post (cols)
    gap_nn = mat_g[np.ix_(idx_r_g, idx_c_g)]    # symmetric

    # Save cache
    try:
        np.savez_compressed(
            str(cache_path),
            mtime=float(mtime) if mtime is not None else -1.0,
            neurons=np.array(neurons, dtype=object),
            chem_nn=chem_nn,
            gap_nn=gap_nn,
        )
    except Exception:
        pass

    return xlsx_path, neurons, chem_nn, gap_nn


def _motor_groups(neurons):
    # Very simple proxy: forward motor ~ VB/DB, backward motor ~ VA/DA.
    fwd = []
    bwd = []
    for i, n in enumerate(neurons):
        if re.match(r"^(VB|DB)\d+$", n):
            fwd.append(i)
        if re.match(r"^(VA|DA)\d+$", n):
            bwd.append(i)
    return fwd, bwd


def _iter_atlas_dirs():
    env_path = os.environ.get("CE_ATLAS_DIR", "").strip()
    if env_path:
        yield env_path
    # Common local fallbacks for manual runs
    yield "data"
    yield "input/data"
    # Repo-local absolute fallback
    yield "/home/rag/Projects/science-codeevolve/problems/elegans/data"


def _align_matrix(src_neurons, mat, target_neurons):
    idx = {n: i for i, n in enumerate(src_neurons)}
    n = len(target_neurons)
    out = np.zeros((n, n), dtype=float)
    src_idx = [idx.get(nm) for nm in target_neurons]
    valid = [i for i, si in enumerate(src_idx) if si is not None]
    if valid:
        sel = [src_idx[i] for i in valid]
        out[np.ix_(valid, valid)] = mat[np.ix_(sel, sel)]
    return out


def _load_neuropeptide_connectome(neurons):
    for d in _iter_atlas_dirs():
        npz_path = Path(d) / "neuropeptide_connectome_aligned.npz"
        if npz_path.exists():
            try:
                data = np.load(str(npz_path), allow_pickle=True)
                if "neurons" in data and "pep_adj" in data:
                    src = data["neurons"].tolist()
                    mat = data["pep_adj"]
                    if len(src) == len(neurons) and list(src) == list(neurons):
                        return mat
                    return _align_matrix(src, mat, neurons)
            except Exception:
                pass

        csv_path = Path(d) / "neuropeptide_connectome_short_range.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                if df.shape[1] < 2 or df.columns[0] != "Row":
                    continue
                rows = df["Row"].astype(str).tolist()
                cols = [c for c in df.columns[1:]]
                mat = df.iloc[:, 1:].to_numpy(dtype=float)
                row_index = {n: i for i, n in enumerate(rows)}
                col_index = {n: i for i, n in enumerate(cols)}
                n = len(neurons)
                out = np.zeros((n, n), dtype=float)
                idx_r = [row_index.get(nm) for nm in neurons]
                idx_c = [col_index.get(nm) for nm in neurons]
                valid_r = [i for i, ri in enumerate(idx_r) if ri is not None]
                valid_c = [j for j, cj in enumerate(idx_c) if cj is not None]
                if valid_r and valid_c:
                    out[np.ix_(valid_r, valid_c)] = mat[
                        np.ix_([idx_r[i] for i in valid_r], [idx_c[j] for j in valid_c])
                    ]
                return out
            except Exception:
                pass
    return None


def _load_cengen_peptide_expr(neurons):
    for d in _iter_atlas_dirs():
        npz_path = Path(d) / "cengen_peptide_expr_by_neuron.npz"
        if npz_path.exists():
            try:
                data = np.load(str(npz_path), allow_pickle=True)
                if "neurons" in data and "pep_expr" in data:
                    src = data["neurons"].tolist()
                    expr = data["pep_expr"]
                    idx = {n: i for i, n in enumerate(src)}
                    out = np.zeros(len(neurons), dtype=float)
                    for i, nm in enumerate(neurons):
                        j = idx.get(nm)
                        if j is not None:
                            out[i] = float(expr[j])
                    if np.any(out):
                        return out
            except Exception:
                pass
    return None


def simulate_c_elegans():
    seed = int(os.environ.get("CE_SEED", "42"))
    rng = np.random.default_rng(seed)

    xlsx_path, neurons, chem_nn, gap_nn = load_connectome_neuron_submatrices()
    n = len(neurons)

    # Normalize chemical by outgoing strength to keep dynamics stable-ish
    out_strength = np.maximum(1.0, chem_nn.sum(axis=1, keepdims=True))
    W = chem_nn / out_strength  # rows sum <= 1

    # Gap coupling as Laplacian term
    G = gap_nn
    deg = np.sum(G, axis=1)
    L = np.diag(deg) - G

    # Optional neuropeptide signaling (atlas-driven if available)
    pep_expr = _load_cengen_peptide_expr(neurons)
    if pep_expr is not None:
        pep_expr = pep_expr / (float(np.max(pep_expr)) + 1e-8)
    pep_adj = _load_neuropeptide_connectome(neurons)
    if pep_adj is not None:
        pep_out = np.maximum(1.0, pep_adj.sum(axis=1, keepdims=True))
        P = pep_adj / pep_out
    else:
        P = None

    # State: simple rate model in [0,1]
    a = rng.random(n)

    # Simulation settings
    steps = 400
    dt = 0.1
    segs = 10
    k_neural = 32

    # Parameters (kept mild; evolution can tune inside block)
    alpha = 0.55     # leak/memory
    beta = 0.75      # chemical drive
    gamma = 0.015    # gap diffusion strength
    bias = 0.02

    pep_gain = 0.04       # neuropeptide modulation strength
    pep_bias_gain = 0.02  # baseline peptide bias from expression
    tau_pep = 2.5         # slow time constant

    base_speed = 0.12
    speed_gain = 0.10
    omega = 0.4
    omega_gain = 0.8
    curv_amp = 0.35
    curv_drive_gain = 0.20
    turn_gain = 0.6

    fwd_idx, bwd_idx = _motor_groups(neurons)
    if len(fwd_idx) == 0 or len(bwd_idx) == 0:
        fwd_idx = list(range(0, min(10, n)))
        bwd_idx = list(range(min(10, n), min(20, n)))

    phase = 0.0
    theta = 0.0
    pos = np.zeros(2, dtype=float)

    positions = np.zeros((steps, 2), dtype=float)
    velocities = np.zeros((steps, 2), dtype=float)
    curvature = np.zeros((steps, segs), dtype=float)
    neural_out = np.zeros((steps, k_neural), dtype=float)

    phase_offsets = np.linspace(0.0, 2.0 * np.pi, segs, endpoint=False)

    m = np.zeros(n, dtype=float)

    for t in range(steps):
        chem_drive = W.T @ a
        gap_drive = -L @ a  # diffusion
        x = beta * chem_drive + gamma * gap_drive + bias
        if pep_expr is not None:
            x = x + pep_bias_gain * pep_expr
        if P is not None:
            m += (P @ a - m) * (dt / tau_pep)
            mod = m
            if pep_expr is not None:
                mod = mod * (1.0 + pep_expr)
            x = x + pep_gain * mod
        a = alpha * a + (1.0 - alpha) * (1.0 / (1.0 + np.exp(-4.0 * (x - 0.5))))

        drive = float(np.mean(a[fwd_idx]) - np.mean(a[bwd_idx]))
        phase += omega + omega_gain * drive

        wave = np.sin(phase + phase_offsets)
        curv = curv_amp * wave + curv_drive_gain * drive
        curvature[t] = curv

        theta += turn_gain * float(curv.mean()) * dt

        speed = float(np.clip(base_speed + speed_gain * np.tanh(3.0 * drive), 0.0, 0.4))
        v = speed * np.array([np.cos(theta), np.sin(theta)])

        pos = pos + v * dt
        positions[t] = pos
        velocities[t] = v

        neural_out[t] = a[:k_neural]

    # Emit connectome stats so evaluator can verify you're really using the file.
    chem_sum = float(chem_nn.sum())
    chem_nnz = int((chem_nn > 0).sum())
    gap_sum = float(gap_nn.sum())
    gap_nnz = int((gap_nn > 0).sum())

    return {
        "positions": positions.tolist(),
        "velocities": velocities.tolist(),
        "curvature": curvature.tolist(),
        "neural": neural_out.tolist(),
        "dt": float(dt),
        "n_neurons": int(n),
        "chem_sum": chem_sum,
        "chem_nnz": chem_nnz,
        "gap_sum": gap_sum,
        "gap_nnz": gap_nnz,
        "source_xlsx": Path(xlsx_path).name,
    }
# EVOLVE-BLOCK-END


if __name__ == "__main__":
    out = simulate_c_elegans()
    print(json.dumps(out))
