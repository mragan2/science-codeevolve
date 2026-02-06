# EVOLVE-BLOCK-START
import os
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---- Connectome loader for SI5 Excel layout ----
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
    candidates = []
    env_path = os.environ.get("CE_CONNECTOME_PATH", "").strip()
    if env_path:
        candidates.append(env_path)

    candidates += [
        "connectome.xlsx",
        "input/connectome.xlsx",
        "SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",
        "/mnt/data/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",
        "/home/rag/Projects/science-codeevolve/problems/elegans/input/connectome.xlsx",
    ]

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
        raise FileNotFoundError("Could not find connectome workbook.")

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
    """Identify forward (VB/DB) and backward (VA/DA) motor neuron groups."""
    fwd = []
    bwd = []
    for i, n in enumerate(neurons):
        if re.match(r"^(VB|DB)\d+$", n):
            fwd.append(i)
        elif re.match(r"^(VA|DA)\d+$", n):
            bwd.append(i)
    return fwd, bwd


def _interneuron_groups(neurons):
    """Identify command interneurons that modulate locomotion."""
    cmd_fwd = []  # AVB class - forward command
    cmd_bwd = []  # AVA class - backward command
    for i, n in enumerate(neurons):
        if re.match(r"^AVB[LR]?$", n):
            cmd_fwd.append(i)
        elif re.match(r"^AVA[LR]?$", n):
            cmd_bwd.append(i)
    return cmd_fwd, cmd_bwd


def _spectral_normalize(W, target_radius=0.95):
    """Normalize weight matrix to have spectral radius <= target_radius."""
    if W.size == 0:
        return W
    eigvals = np.linalg.eigvals(W)
    rho = np.max(np.abs(eigvals))
    if rho > 1e-10:
        W = W * (target_radius / rho)
    return W


def simulate_c_elegans():
    seed = int(os.environ.get("CE_SEED", "42"))
    rng = np.random.default_rng(seed)

    xlsx_path, neurons, chem_nn, gap_nn = load_connectome_neuron_submatrices()
    n = len(neurons)

    # ========== WEIGHT MATRIX PREPARATION ==========
    # Chemical synapse weights with spectral normalization
    out_strength = np.maximum(1.0, chem_nn.sum(axis=1, keepdims=True))
    W_chem = chem_nn / out_strength
    W_chem = _spectral_normalize(W_chem, target_radius=0.92)

    # Gap junction coupling (symmetric diffusive coupling)
    gap_degree = np.maximum(1.0, gap_nn.sum(axis=1, keepdims=True))
    G = gap_nn / gap_degree
    L_gap = np.diag(np.sum(G, axis=1)) - G  # Laplacian

    # ========== NEURAL DYNAMICS PARAMETERS ==========
    # LIF-like dynamics with adaptation
    tau_m = 0.12          # Membrane time constant
    tau_syn = 0.08        # Synaptic filtering time constant
    tau_adapt = 0.25      # Adaptation time constant
    
    alpha_leak = 0.55     # Leak/decay rate
    beta_chem = 0.68      # Chemical synapse strength
    gamma_gap = 0.022     # Gap junction coupling strength
    
    noise_std = 0.035     # Neural noise level
    adapt_strength = 0.15 # Spike-rate adaptation strength
    
    # Activation function parameters
    gain = 4.2            # Sigmoid gain
    threshold = 0.48      # Sigmoid threshold
    
    # ========== KINEMATICS PARAMETERS ==========
    steps = 400
    dt = 0.1
    segs = 10
    
    # Movement parameters
    base_speed = 0.12
    speed_gain = 0.14
    max_speed = 0.45
    
    # Oscillation parameters (traveling wave)
    omega_base = 0.42     # Base angular frequency
    omega_gain = 0.55     # Frequency modulation by drive
    
    # Curvature parameters (critical for fitness!)
    curv_amp_base = 0.38  # Base curvature amplitude
    curv_amp_gain = 0.18  # Amplitude modulation by drive
    curv_prop_fb = 0.12   # Proprioceptive feedback strength
    
    # Turning dynamics
    turn_gain = 0.48
    turn_damping = 0.92
    
    # Body mechanics
    segment_coupling = 0.08  # Mechanical coupling between segments
    
    # ========== MOTOR GROUPS ==========
    fwd_idx, bwd_idx = _motor_groups(neurons)
    cmd_fwd_idx, cmd_bwd_idx = _interneuron_groups(neurons)
    
    # Fallback if motor neurons not found
    if len(fwd_idx) == 0 or len(bwd_idx) == 0:
        fwd_idx = list(range(0, min(10, n)))
        bwd_idx = list(range(min(10, n), min(20, n)))
    
    # Weights for motor neuron contributions (posterior neurons have more weight)
    fwd_weights = np.array([1.0 + 0.1 * i for i in range(len(fwd_idx))])
    fwd_weights /= fwd_weights.sum()
    bwd_weights = np.array([1.0 + 0.1 * i for i in range(len(bwd_idx))])
    bwd_weights /= bwd_weights.sum()

    # ========== STATE INITIALIZATION ==========
    # Neural state
    a = rng.random(n) * 0.3 + 0.2  # Activity [0.2, 0.5]
    s = np.zeros(n)                 # Synaptic variable
    w_adapt = np.zeros(n)           # Adaptation variable
    
    # Kinematics state
    phase = rng.random() * 2 * np.pi  # Random initial phase
    theta = rng.random() * 2 * np.pi  # Random heading
    omega_theta = 0.0                  # Angular velocity
    pos = np.zeros(2, dtype=float)
    
    # Body segment state
    segment_phase = np.zeros(segs)
    segment_curv = np.zeros(segs)
    
    # Drive signal smoothing (EMA)
    drive_smooth = 0.0
    drive_ema_alpha = 0.15
    
    # ========== OUTPUT ARRAYS ==========
    positions = np.zeros((steps, 2), dtype=float)
    velocities = np.zeros((steps, 2), dtype=float)
    curvature = np.zeros((steps, segs), dtype=float)
    neural_out = np.zeros((steps, 32), dtype=float)
    
    # Phase offsets: nonlinear (head leads, propagates posteriorly)
    # Real C. elegans has ~1.5 wavelengths along body
    phase_offsets = np.zeros(segs)
    for i in range(segs):
        # Nonlinear spacing: tighter at head, spreads at tail
        frac = i / (segs - 1)
        phase_offsets[i] = 1.5 * np.pi * (frac ** 0.85)
    
    # ========== MAIN SIMULATION LOOP ==========
    for t in range(steps):
        # --- Neural dynamics update ---
        # Synaptic input with temporal filtering
        chem_input = W_chem.T @ a
        s = s + dt / tau_syn * (-s + chem_input)
        
        # Gap junction coupling (diffusive)
        gap_input = -L_gap @ a
        
        # Total input current
        I_total = beta_chem * s + gamma_gap * gap_input
        
        # Add noise
        noise = rng.normal(0, noise_std, n)
        I_total = I_total + noise
        
        # Membrane potential update with leak and adaptation
        da = dt / tau_m * (-alpha_leak * a + I_total - adapt_strength * w_adapt)
        a = a + da
        
        # Activation function (soft rectification) with bounded input
        z = gain * (a - threshold)
        z = np.clip(z, -10.0, 10.0)
        a = 1.0 / (1.0 + np.exp(-z))
        
        # Adaptation update (slow negative feedback)
        w_adapt = w_adapt + dt / tau_adapt * (-w_adapt + a)
        
        # Clip to valid range
        a = np.clip(a, 0.0, 1.0)
        
        # --- Motor drive computation ---
        # Weighted motor neuron activity
        fwd_act = np.sum(a[fwd_idx] * fwd_weights) if len(fwd_idx) > 0 else 0.0
        bwd_act = np.sum(a[bwd_idx] * bwd_weights) if len(bwd_idx) > 0 else 0.0
        
        # Command interneuron contribution
        if len(cmd_fwd_idx) > 0:
            fwd_act += 0.3 * np.mean(a[cmd_fwd_idx])
        if len(cmd_bwd_idx) > 0:
            bwd_act += 0.3 * np.mean(a[cmd_bwd_idx])
        
        # Raw drive signal
        drive_raw = np.tanh(2.5 * (fwd_act - bwd_act))
        
        # Smooth drive with EMA
        drive_smooth = drive_ema_alpha * drive_raw + (1 - drive_ema_alpha) * drive_smooth
        
        # --- Phase and frequency update ---
        omega = omega_base + omega_gain * drive_smooth
        phase += omega * dt
        
        # --- Body curvature computation (critical for fitness!) ---
        # Amplitude varies with drive magnitude
        curv_amp = curv_amp_base + curv_amp_gain * np.abs(drive_smooth)
        
        # Traveling wave with segment-specific phases
        wave = np.sin(phase - phase_offsets)
        
        # Proprioceptive feedback: each segment influenced by neighbors
        prop_feedback = np.zeros(segs)
        for i in range(segs):
            if i > 0:
                prop_feedback[i] += curv_prop_fb * segment_curv[i-1]
            if i < segs - 1:
                prop_feedback[i] += curv_prop_fb * segment_curv[i+1]
        
        # Mechanical coupling smoothing
        for i in range(1, segs):
            wave[i] = (1 - segment_coupling) * wave[i] + segment_coupling * wave[i-1]
        
        # Final curvature with drive modulation and proprioceptive feedback
        segment_curv = curv_amp * wave + 0.5 * curv_amp_gain * drive_smooth + prop_feedback
        
        # Add small noise for variability matching real data
        segment_curv += rng.normal(0, 0.02, segs)
        
        curvature[t] = segment_curv
        
        # --- Turning and heading update ---
        # Heading change from body curvature (weighted toward head segments)
        head_curv = np.mean(segment_curv[:3])  # Head segments
        omega_theta = turn_damping * omega_theta + turn_gain * head_curv * dt
        theta += omega_theta * dt
        
        # --- Speed and position update ---
        # Speed depends on forward drive (positive = forward)
        speed = np.clip(base_speed + speed_gain * drive_smooth, 0.02, max_speed)
        
        # Velocity vector
        v = speed * np.array([np.cos(theta), np.sin(theta)])
        
        # Position update
        pos = pos + v * dt
        
        # --- Store outputs ---
        positions[t] = pos
        velocities[t] = v
        neural_out[t] = a[:32]
    
    # ========== CONNECTOME STATS FOR VALIDATION ==========
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
