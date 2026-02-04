#!/usr/bin/env python
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import requests  # type: ignore
except Exception:
    requests = None
    from urllib.request import urlopen

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / ".." / "input"
CONNECTOME_PATH = (INPUT_DIR / "connectome.xlsx").resolve()

CENGEN_URL = "https://cengen.org/storage/021821_medium_threshold2.csv"
NEUROPEP_URL = (
    "https://raw.githubusercontent.com/LidiaRipollSanchez/Neuropeptide-Connectome/main/"
    "Adjacency%20matrices%20for%20networks/01022024_neuropeptide_connectome_short_range_model.csv"
)

CENGEN_CSV = BASE_DIR / "cengen_medium_threshold2.csv"
NEUROPEP_CSV = BASE_DIR / "neuropeptide_connectome_short_range.csv"
CENGEN_NPZ = BASE_DIR / "cengen_peptide_expr_by_neuron.npz"
NEUROPEP_NPZ = BASE_DIR / "neuropeptide_connectome_aligned.npz"
META_JSON = BASE_DIR / "atlas_sources.json"


def _download(url: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if requests is not None:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
        return
    with urlopen(url, timeout=120) as resp:
        if getattr(resp, "status", 200) >= 400:
            raise RuntimeError(f"Download failed with status {resp.status} for {url}")
        path.write_bytes(resp.read())


def _clean_label(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    return s if s else None


def _truncate(labels):
    last = -1
    for i, x in enumerate(labels):
        if x is not None:
            last = i
    return labels[: last + 1]


def _load_si5_sheet_matrix(xlsx_path: Path, sheet_name: str):
    raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)
    col_labels = [_clean_label(x) for x in raw.iloc[2, 3:].tolist()]
    row_labels = [_clean_label(x) for x in raw.iloc[3:, 2].tolist()]
    col_labels = _truncate(col_labels)
    row_labels = _truncate(row_labels)
    mat = raw.iloc[3 : 3 + len(row_labels), 3 : 3 + len(col_labels)]
    mat = mat.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return row_labels, col_labels, mat


def _load_connectome_neurons():
    if not CONNECTOME_PATH.exists():
        raise FileNotFoundError(f"Connectome not found: {CONNECTOME_PATH}")
    r_c, c_c, _ = _load_si5_sheet_matrix(CONNECTOME_PATH, "hermaphrodite chemical")
    neuron_set = set(r_c) & set(c_c)
    neurons = [n for n in r_c if n in neuron_set]
    return neurons


def _map_cengen_col(neuron: str, col_set):
    if neuron in col_set:
        return neuron

    # Strip L/R suffix
    if neuron.endswith("L") or neuron.endswith("R"):
        base = neuron[:-1]
        if base in col_set:
            return base

    # Handle numbered neurons: VB01 -> VB
    m = re.match(r"^([A-Z]+)", neuron)
    if m:
        prefix = m.group(1)
        if prefix in col_set:
            return prefix
        if prefix in {"VD", "DD"} and "VD_DD" in col_set:
            return "VD_DD"

    return None


def _build_peptide_expression(neurons):
    df = pd.read_csv(CENGEN_CSV)
    if "gene_name" not in df.columns:
        raise ValueError("CeNGEN CSV missing gene_name column")

    pep_mask = df["gene_name"].astype(str).str.match(r"^(flp|nlp|ins|pdf)-", case=False)
    pep_df = df.loc[pep_mask]

    meta_cols = {"Unnamed: 0", "gene_name", "Wormbase_ID"}
    neuron_cols = [c for c in df.columns if c not in meta_cols]
    if not neuron_cols:
        raise ValueError("No neuron columns found in CeNGEN CSV")

    col_expr = pep_df[neuron_cols].sum(axis=0)
    col_set = set(col_expr.index)

    pep_expr = np.zeros(len(neurons), dtype=float)
    for i, n in enumerate(neurons):
        col = _map_cengen_col(n, col_set)
        if col is not None:
            pep_expr[i] = float(col_expr[col])

    return pep_expr


def _build_neuropeptide_connectome(neurons):
    df = pd.read_csv(NEUROPEP_CSV)
    if df.shape[1] < 2 or df.columns[0] != "Row":
        raise ValueError("Unexpected neuropeptide connectome CSV format")

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


def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading CeNGEN expression atlas...")
    _download(CENGEN_URL, CENGEN_CSV)
    print("Downloading neuropeptide connectome...")
    _download(NEUROPEP_URL, NEUROPEP_CSV)

    neurons = _load_connectome_neurons()

    print("Building peptide expression vector...")
    pep_expr = _build_peptide_expression(neurons)
    np.savez_compressed(CENGEN_NPZ, neurons=np.array(neurons, dtype=object), pep_expr=pep_expr)

    print("Building aligned neuropeptide connectome...")
    pep_adj = _build_neuropeptide_connectome(neurons)
    np.savez_compressed(NEUROPEP_NPZ, neurons=np.array(neurons, dtype=object), pep_adj=pep_adj)

    META_JSON.write_text(
        json.dumps(
            {
                "cengen_url": CENGEN_URL,
                "neuropeptide_connectome_url": NEUROPEP_URL,
                "connectome": str(CONNECTOME_PATH),
            },
            indent=2,
        )
    )

    print("Done.")
    print(f"Saved: {CENGEN_CSV}")
    print(f"Saved: {NEUROPEP_CSV}")
    print(f"Saved: {CENGEN_NPZ}")
    print(f"Saved: {NEUROPEP_NPZ}")


if __name__ == "__main__":
    main()
