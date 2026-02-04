import os
import sys
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import h5py


# ---- Connectome path helpers ----
# We want evaluation and candidate code to reliably find the workbook regardless of CWD.
ELEGANS_INPUT_DIR = Path(__file__).resolve().parent

def _iter_connectome_candidates():
    env_path = os.environ.get("CE_CONNECTOME_PATH", "").strip()
    if env_path:
        yield env_path
    # Prefer the workbook shipped with the problem itself
    yield str(ELEGANS_INPUT_DIR / "connectome.xlsx")
    yield str(ELEGANS_INPUT_DIR / "SI 5 Connectome adjacency matrices, corrected July 2020.xlsx")
    # Common fallbacks (CWD)
    yield "connectome.xlsx"
    yield "SI 5 Connectome adjacency matrices, corrected July 2020.xlsx"
    # Chat / mounted fallback
    yield "/mnt/data/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx"

# --------------------
# Connectome utilities
# --------------------

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


def _load_si5_sheet_matrix(xlsx_path: str, sheet_name: str):
    raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)
    col_labels = [_clean_label(x) for x in raw.iloc[2, 3:].tolist()]
    row_labels = [_clean_label(x) for x in raw.iloc[3:, 2].tolist()]
    col_labels = _truncate(col_labels)
    row_labels = _truncate(row_labels)
    mat = raw.iloc[3 : 3 + len(row_labels), 3 : 3 + len(col_labels)]
    mat = mat.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return row_labels, col_labels, mat


def expected_stats():
    xlsx_path = None
    for c in _iter_connectome_candidates():
        if Path(c).exists():
            xlsx_path = str(Path(c))
            break
    if xlsx_path is None:
        raise FileNotFoundError("Connectome workbook not found for evaluation.")

    cache_path = Path("connectome_cache_eval_stats.json")
    try:
        mtime = Path(xlsx_path).stat().st_mtime
    except Exception:
        mtime = None

    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text())
            if ("mtime" in data) and (mtime is None or float(data["mtime"]) == float(mtime)):
                return data["stats"]
        except Exception:
            pass

    r_c, c_c, mat_c = _load_si5_sheet_matrix(xlsx_path, "hermaphrodite chemical")
    r_g, c_g, mat_g = _load_si5_sheet_matrix(xlsx_path, "hermaphrodite gap jn symmetric")

    neuron_set = set(r_c) & set(c_c)
    neurons = [n for n in r_c if n in neuron_set]

    row_index_c = {n: i for i, n in enumerate(r_c)}
    col_index_c = {n: i for i, n in enumerate(c_c)}
    row_index_g = {n: i for i, n in enumerate(r_g)}
    col_index_g = {n: i for i, n in enumerate(c_g)}

    idx_r_c = [row_index_c[n] for n in neurons]
    idx_c_c = [col_index_c[n] for n in neurons]
    idx_r_g = [row_index_g[n] for n in neurons]
    idx_c_g = [col_index_g[n] for n in neurons]

    chem_nn = mat_c[np.ix_(idx_r_c, idx_c_c)]
    gap_nn = mat_g[np.ix_(idx_r_g, idx_c_g)]

    stats = {
        "n_neurons": int(len(neurons)),
        "chem_sum": float(chem_nn.sum()),
        "chem_nnz": int((chem_nn > 0).sum()),
        "gap_sum": float(gap_nn.sum()),
        "gap_nnz": int((gap_nn > 0).sum()),
    }

    try:
        cache_path.write_text(json.dumps({"mtime": float(mtime) if mtime is not None else -1.0, "stats": stats}))
    except Exception:
        pass

    return stats


# --------------------
# Target data utilities
# --------------------

DATA_DIR = Path(__file__).parent / "data"
DRYAD_DIR = DATA_DIR / "dryad2024"
WW_DIR = DATA_DIR / "wormwideweb" / "processed_h5" / "processed_h5"
CACHE_WW = DATA_DIR / "targets_wormwideweb.npz"
CACHE_DRYAD = DATA_DIR / "targets_dryad2024.npz"

PSD_BINS = 32
NEURAL_K = 32
REQUIRE_DRYAD = os.environ.get("CE_SKIP_DRYAD", "0").lower() not in {"1", "true", "yes"}


def _psd(x, n_bins=PSD_BINS):
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        return np.zeros(n_bins, dtype=float)
    x = x - np.mean(x)
    fft = np.fft.rfft(x)
    power = np.abs(fft) ** 2
    if power.size < n_bins:
        power = np.pad(power, (0, n_bins - power.size), mode="constant")
    else:
        power = power[:n_bins]
    s = power.sum()
    if s > 0:
        power = power / s
    return power.astype(float)


def _norm_rmse(a, b, eps=1e-8, scale=None):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        m = min(a.size, b.size)
        a = a.reshape(-1)[:m]
        b = b.reshape(-1)[:m]
    num = np.sqrt(np.mean((a - b) ** 2))
    if scale is None:
        denom = np.sqrt(np.mean(b ** 2)) + eps
    else:
        s = np.asarray(scale, dtype=float)
        denom = np.sqrt(np.mean(s ** 2)) + eps
    return float(num / denom)


def _downsample_curvature(curv, target_segments=10):
    curv = np.asarray(curv, dtype=float)
    if curv.ndim != 2:
        raise ValueError("curvature must be 2D (T, segments)")
    t, s = curv.shape
    if s == target_segments:
        return curv
    if s % target_segments == 0:
        block = s // target_segments
        return curv.reshape(t, target_segments, block).mean(axis=2)
    # interpolate along segment axis
    xs = np.linspace(0, 1, s)
    xt = np.linspace(0, 1, target_segments)
    out = np.empty((t, target_segments), dtype=float)
    for i in range(t):
        out[i] = np.interp(xt, xs, curv[i])
    return out


def _load_wormwideweb_targets():
    if CACHE_WW.exists():
        try:
            data = np.load(str(CACHE_WW), allow_pickle=True)
            if int(data.get("version", 0)) == 1:
                return {k: data[k] for k in data.files}
        except Exception:
            pass

    if not WW_DIR.exists():
        raise FileNotFoundError("WormWideWeb processed_h5 directory not found.")

    files = sorted(WW_DIR.glob("*.h5"))
    if not files:
        raise FileNotFoundError("No WormWideWeb .h5 files found.")

    # Use a small subset for speed; still representative
    files = files[:3]

    curv_mean = np.zeros(10, dtype=float)
    curv_std = np.zeros(10, dtype=float)
    curv_psd = np.zeros(PSD_BINS, dtype=float)

    vel_mean = 0.0
    vel_std = 0.0
    vel_psd = np.zeros(PSD_BINS, dtype=float)

    traj_speed_mean = 0.0
    traj_speed_std = 0.0

    neural_mean = np.zeros(NEURAL_K, dtype=float)
    neural_std = np.zeros(NEURAL_K, dtype=float)
    neural_corr = np.zeros((NEURAL_K, NEURAL_K), dtype=float)

    dt_vals = []

    for fp in files:
        with h5py.File(fp, "r") as f:
            vel = f["behavior/velocity"][()]
            body_angle = f["behavior/body_angle"][()]
            stage_x = f["behavior/stage_x"][()]
            stage_y = f["behavior/stage_y"][()]
            trace = f["gcamp/trace_array"][()]
            t_conf = f["timing/timestamp_confocal"][()]

        if t_conf.size > 1:
            dt = float(np.median(np.diff(t_conf)))
            if np.isfinite(dt) and dt > 0:
                dt_vals.append(dt)

        # Curvature (downsample to 10 segments)
        curv10 = _downsample_curvature(body_angle, target_segments=10)
        curv_mean += curv10.mean(axis=0)
        curv_std += curv10.std(axis=0)
        curv_psd += _psd(curv10.mean(axis=1))

        # Velocity
        vel = np.asarray(vel, dtype=float)
        vel_mean += float(np.mean(vel))
        vel_std += float(np.std(vel))
        vel_psd += _psd(vel)

        # Trajectory speed from stage positions
        pos = np.stack([stage_x, stage_y], axis=1)
        if pos.shape[0] > 1:
            if dt_vals:
                dt_use = dt_vals[-1]
            else:
                dt_use = 1.0
            traj_speed = np.linalg.norm(np.diff(pos, axis=0) / dt_use, axis=1)
            traj_speed_mean += float(np.mean(traj_speed))
            traj_speed_std += float(np.std(traj_speed))

        # Neural
        trace = np.asarray(trace, dtype=float)
        if trace.shape[1] < NEURAL_K:
            raise ValueError("WormWideWeb trace_array has fewer neurons than required.")
        neural = trace[:, :NEURAL_K]
        neural_mean += neural.mean(axis=0)
        neural_std += neural.std(axis=0)
        corr = np.corrcoef(neural, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        neural_corr += corr

    n = len(files)
    curv_mean /= n
    curv_std /= n
    curv_psd /= max(n, 1)
    vel_mean /= n
    vel_std /= n
    vel_psd /= max(n, 1)
    traj_speed_mean /= n
    traj_speed_std /= n
    neural_mean /= n
    neural_std /= n
    neural_corr /= n

    dt_target = float(np.median(dt_vals)) if dt_vals else 1.0

    payload = {
        "version": 1,
        "dt": dt_target,
        "curv_mean": curv_mean,
        "curv_std": curv_std,
        "curv_psd": curv_psd,
        "vel_mean": vel_mean,
        "vel_std": vel_std,
        "vel_psd": vel_psd,
        "traj_speed_mean": traj_speed_mean,
        "traj_speed_std": traj_speed_std,
        "neural_mean": neural_mean,
        "neural_std": neural_std,
        "neural_corr": neural_corr,
        "neural_k": NEURAL_K,
    }

    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(CACHE_WW), **payload)
    except Exception:
        pass

    return payload


def _extract_dryad_zip(zip_path: Path, out_dir: Path):
    import zipfile

    out_dir.mkdir(parents=True, exist_ok=True)
    if not zipfile.is_zipfile(str(zip_path)):
        raise FileNotFoundError(
            "dryad.zip is not a valid zip file. Please download the dataset via browser "
            "and place it at problems/elegans/input/data/dryad2024/dryad.zip."
        )
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(out_dir))


def _find_dryad_files(root: Path):
    files = []
    for ext in ("*.txt", "*.csv", "*.tsv"):
        files.extend(root.rglob(ext))
    files = [f for f in files if "readme" not in f.name.lower()]
    return sorted(files)


def _load_dryad_targets():
    if CACHE_DRYAD.exists():
        try:
            data = np.load(str(CACHE_DRYAD), allow_pickle=True)
            if int(data.get("version", 0)) == 1:
                return {k: data[k] for k in data.files}
        except Exception:
            pass

    if not DRYAD_DIR.exists():
        raise FileNotFoundError("Dryad directory not found.")

    zip_path = DRYAD_DIR / "dryad.zip"
    extract_dir = DRYAD_DIR / "extracted"

    if zip_path.exists() and not extract_dir.exists():
        _extract_dryad_zip(zip_path, extract_dir)

    search_root = extract_dir if extract_dir.exists() else DRYAD_DIR
    files = _find_dryad_files(search_root)
    if not files:
        raise FileNotFoundError("No Dryad curvature files found.")

    curv_mean = np.zeros(10, dtype=float)
    curv_std = np.zeros(10, dtype=float)
    curv_psd = np.zeros(PSD_BINS, dtype=float)
    dt_vals = []

    used = 0
    for fp in files:
        try:
            data = np.genfromtxt(fp, delimiter=None)
        except Exception:
            continue
        if data.ndim != 2 or data.shape[0] < 5:
            continue
        if data.shape[1] >= 11:
            t = data[:, 0]
            curv = data[:, 1:11]
        elif data.shape[1] >= 10:
            t = None
            curv = data[:, :10]
        else:
            continue

        curv = np.asarray(curv, dtype=float)
        curv = _downsample_curvature(curv, target_segments=10)
        curv_mean += curv.mean(axis=0)
        curv_std += curv.std(axis=0)
        curv_psd += _psd(curv.mean(axis=1))

        if t is not None and len(t) > 1:
            dt = float(np.median(np.diff(t)))
            if np.isfinite(dt) and dt > 0:
                dt_vals.append(dt)
        used += 1

    if used == 0:
        raise FileNotFoundError("Dryad files found but none could be parsed.")

    curv_mean /= used
    curv_std /= used
    curv_psd /= max(used, 1)
    dt_target = float(np.median(dt_vals)) if dt_vals else 1.0

    payload = {
        "version": 1,
        "dt": dt_target,
        "curv_mean": curv_mean,
        "curv_std": curv_std,
        "curv_psd": curv_psd,
    }

    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(CACHE_DRYAD), **payload)
    except Exception:
        pass

    return payload


def _compute_candidate_features(out, target):
    positions = np.asarray(out.get("positions"), dtype=float)
    velocities = np.asarray(out.get("velocities"), dtype=float)
    curvature = np.asarray(out.get("curvature"), dtype=float)
    neural = np.asarray(out.get("neural"), dtype=float)

    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must be shape (T, 2)")
    if velocities.ndim != 2 or velocities.shape[1] != 2:
        raise ValueError("velocities must be shape (T, 2)")
    if curvature.ndim != 2:
        raise ValueError("curvature must be shape (T, segments)")
    if neural.ndim != 2:
        raise ValueError("neural must be shape (T, K)")

    t = min(len(positions), len(velocities), len(curvature), len(neural))
    positions = positions[:t]
    velocities = velocities[:t]
    curvature = curvature[:t]
    neural = neural[:t]

    # Kinematics
    speed = np.linalg.norm(velocities, axis=1)
    curv10 = _downsample_curvature(curvature, target_segments=10)
    curv_mean = curv10.mean(axis=0)
    curv_std = curv10.std(axis=0)
    curv_psd = _psd(curv10.mean(axis=1))

    vel_mean = float(np.mean(speed))
    vel_std = float(np.std(speed))
    vel_psd = _psd(speed)

    dt = float(out.get("dt", target.get("dt", 1.0)))
    if not np.isfinite(dt) or dt <= 0:
        dt = float(target.get("dt", 1.0))

    if positions.shape[0] > 1:
        traj_speed = np.linalg.norm(np.diff(positions, axis=0) / dt, axis=1)
        traj_speed_mean = float(np.mean(traj_speed))
        traj_speed_std = float(np.std(traj_speed))
    else:
        traj_speed_mean = 0.0
        traj_speed_std = 0.0

    # Neural
    k_target = int(target.get("neural_k", NEURAL_K))
    if neural.shape[1] < k_target:
        raise ValueError("neural output has fewer channels than required")
    neural = neural[:, :k_target]
    neural_mean = neural.mean(axis=0)
    neural_std = neural.std(axis=0)
    neural_corr = np.corrcoef(neural, rowvar=False)
    neural_corr = np.nan_to_num(neural_corr, nan=0.0, posinf=0.0, neginf=0.0)

    # Consistency penalty (positions vs velocities)
    if positions.shape[0] > 1:
        vel_from_pos = np.diff(positions, axis=0) / dt
        vel_cmp = velocities[: vel_from_pos.shape[0]]
        consistency = _norm_rmse(vel_cmp, vel_from_pos)
    else:
        consistency = 0.0

    return {
        "curv_mean": curv_mean,
        "curv_std": curv_std,
        "curv_psd": curv_psd,
        "vel_mean": vel_mean,
        "vel_std": vel_std,
        "vel_psd": vel_psd,
        "traj_speed_mean": traj_speed_mean,
        "traj_speed_std": traj_speed_std,
        "neural_mean": neural_mean,
        "neural_std": neural_std,
        "neural_corr": neural_corr,
        "consistency": consistency,
    }


# --------------------
# Evaluation entrypoint
# --------------------

def evaluate(code_path, results_path):
    try:
        exp = expected_stats()
        seeds = [0, 1, 2]
        metrics = []
        last_out = None

        ww_targets = _load_wormwideweb_targets()
        if REQUIRE_DRYAD:
            dryad_targets = _load_dryad_targets()
        else:
            dryad_targets = None

        for s in seeds:
            env = dict(**__import__("os").environ)
            env["CE_SEED"] = str(s)
            # Pass connectome path to candidate (so it doesn't depend on CWD)
            local_conn = ELEGANS_INPUT_DIR / "connectome.xlsx"
            if local_conn.exists():
                env["CE_CONNECTOME_PATH"] = str(local_conn)
            r = subprocess.run(
                [sys.executable, code_path],
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            )
            if r.returncode != 0:
                raise RuntimeError(r.stderr[-2000:])

            out = json.loads(r.stdout)
            last_out = out

            # Enforce connectome-derived stats
            if int(out.get("n_neurons", -1)) != exp["n_neurons"]:
                raise ValueError("n_neurons mismatch")
            if abs(float(out.get("chem_sum", -1.0)) - exp["chem_sum"]) > 1e-6:
                raise ValueError("chem_sum mismatch")
            if int(out.get("chem_nnz", -1)) != exp["chem_nnz"]:
                raise ValueError("chem_nnz mismatch")
            if abs(float(out.get("gap_sum", -1.0)) - exp["gap_sum"]) > 1e-6:
                raise ValueError("gap_sum mismatch")
            if int(out.get("gap_nnz", -1)) != exp["gap_nnz"]:
                raise ValueError("gap_nnz mismatch")

            feat = _compute_candidate_features(out, ww_targets)
            metrics.append(feat)

        # Aggregate across seeds
        def avg(key):
            return np.mean([m[key] for m in metrics], axis=0)

        cand = {
            "curv_mean": avg("curv_mean"),
            "curv_std": avg("curv_std"),
            "curv_psd": avg("curv_psd"),
            "vel_mean": float(np.mean([m["vel_mean"] for m in metrics])),
            "vel_std": float(np.mean([m["vel_std"] for m in metrics])),
            "vel_psd": avg("vel_psd"),
            "traj_speed_mean": float(np.mean([m["traj_speed_mean"] for m in metrics])),
            "traj_speed_std": float(np.mean([m["traj_speed_std"] for m in metrics])),
            "neural_mean": avg("neural_mean"),
            "neural_std": avg("neural_std"),
            "neural_corr": avg("neural_corr"),
            "consistency": float(np.mean([m["consistency"] for m in metrics])),
        }

        # Behavior loss (WormWideWeb targets)
        behavior_loss = (
            0.35 * _norm_rmse(cand["curv_mean"], ww_targets["curv_mean"], scale=ww_targets["curv_std"])
            + 0.25 * _norm_rmse(cand["curv_std"], ww_targets["curv_std"])
            + 0.20 * _norm_rmse(cand["curv_psd"], ww_targets["curv_psd"])
            + 0.10 * _norm_rmse(cand["vel_mean"], ww_targets["vel_mean"], scale=ww_targets["vel_std"])
            + 0.05 * _norm_rmse(cand["vel_std"], ww_targets["vel_std"])
            + 0.05 * _norm_rmse(cand["vel_psd"], ww_targets["vel_psd"])
            + 0.05 * _norm_rmse(
                cand["traj_speed_mean"], ww_targets["traj_speed_mean"], scale=ww_targets["traj_speed_std"]
            )
            + 0.05 * _norm_rmse(cand["traj_speed_std"], ww_targets["traj_speed_std"])
        )

        # Dryad curvature loss (if available)
        if dryad_targets is not None:
            dryad_loss = (
                0.5 * _norm_rmse(cand["curv_mean"], dryad_targets["curv_mean"], scale=dryad_targets["curv_std"])
                + 0.3 * _norm_rmse(cand["curv_std"], dryad_targets["curv_std"])
                + 0.2 * _norm_rmse(cand["curv_psd"], dryad_targets["curv_psd"])
            )
            behavior_loss = 0.6 * behavior_loss + 0.4 * dryad_loss
        else:
            dryad_loss = None

        # Neural loss
        neural_loss = (
            0.4 * _norm_rmse(cand["neural_mean"], ww_targets["neural_mean"], scale=ww_targets["neural_std"])
            + 0.2 * _norm_rmse(cand["neural_std"], ww_targets["neural_std"])
            + 0.4 * _norm_rmse(cand["neural_corr"], ww_targets["neural_corr"])
        )

        consistency_penalty = cand["consistency"]

        total_loss = behavior_loss + neural_loss + 0.1 * consistency_penalty
        # Convert loss to a positive fitness in (0, 1]; higher is better.
        fitness = 1.0 / (1.0 + total_loss)

        last_summary = None
        if isinstance(last_out, dict):
            try:
                last_summary = {
                    "positions_len": len(last_out.get("positions", [])),
                    "velocities_len": len(last_out.get("velocities", [])),
                    "curvature_len": len(last_out.get("curvature", [])),
                    "neural_len": len(last_out.get("neural", [])),
                    "dt": float(last_out.get("dt", 0.0)) if last_out.get("dt") is not None else None,
                }
            except Exception:
                last_summary = None

        payload = {
            "fitness": float(fitness),
            "total_loss": float(total_loss),
            "behavior_loss": float(behavior_loss),
            "neural_loss": float(neural_loss),
            "consistency_penalty": float(consistency_penalty),
            "dryad_loss": float(dryad_loss) if dryad_loss is not None else None,
            "expected": exp,
            "last_output_summary": last_summary,
        }

    except Exception as e:
        payload = {"fitness": -float("inf"), "error": str(e)}

    with open(results_path, "w") as f:
        json.dump(payload, f)


if __name__ == "__main__":
    evaluate(sys.argv[1], sys.argv[2])
