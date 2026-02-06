# EVOLVE-BLOCK-START
import os
import re
import json
from pathlib import Path
import xml.etree.ElementTree as ET

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


def _iter_neuroml_candidates():
    env_path = os.environ.get("CE_NEUROML_PATH", "").strip()
    if env_path:
        yield env_path
    # Prefer explicit atlas/data root (set by elegans.sh)
    atlas_dir = os.environ.get("CE_ATLAS_DIR", "").strip()
    if atlas_dir:
        base = Path(atlas_dir)
        yield str(base / "openworm_neuroml" / "CElegans.net.nml")
        yield str(base / "openworm_neuroml" / "CElegans.nml")
    # Local problem data dir (works when running in-place)
    base = Path(__file__).resolve().parents[1] / "data"
    yield str(base / "openworm_neuroml" / "CElegans.net.nml")
    yield str(base / "openworm_neuroml" / "CElegans.nml")
    # Repo-relative fallback from CWD (works when code is copied to /tmp)
    try:
        cwd = Path.cwd().resolve()
        for parent in [cwd] + list(cwd.parents):
            yield str(parent / "problems" / "elegans" / "input" / "data" / "openworm_neuroml" / "CElegans.net.nml")
            yield str(parent / "problems" / "elegans" / "input" / "data" / "openworm_neuroml" / "CElegans.nml")
    except Exception:
        pass
    # Common repo locations
    yield "/home/rag/Projects/CElegansNeuroML/generatedNeuroML2/CElegans.net.nml"
    yield "/home/rag/Projects/CElegansNeuroML/generatedNeuroML/CElegans.net.nml"


def _load_neuroml_positions(neurons):
    x = None
    for c in _iter_neuroml_candidates():
        if Path(c).exists():
            x = str(Path(c))
            break
    if x is None:
        raise FileNotFoundError(
            "NeuroML file with 3D neuron positions not found. "
            "Set CE_NEUROML_PATH or place CElegans.net.nml in problems/elegans/input/data/openworm_neuroml/."
        )

    tree = ET.parse(x)
    root = tree.getroot()

    def strip_ns(tag):
        return tag.split("}", 1)[-1] if "}" in tag else tag

    def norm_name(name):
        m = re.match(r"^([A-Za-z]+)0+([0-9]+)$", name)
        if m:
            return f"{m.group(1)}{int(m.group(2))}"
        return name

    positions = {}
    for pop in root.iter():
        if strip_ns(pop.tag) != "population":
            continue
        pop_id = pop.attrib.get("id", "")
        for inst in pop:
            if strip_ns(inst.tag) != "instance":
                continue
            loc = None
            for child in inst:
                if strip_ns(child.tag) == "location":
                    loc = child
                    break
            if loc is None:
                continue
            try:
                x_val = float(loc.attrib.get("x", "0"))
                y_val = float(loc.attrib.get("y", "0"))
                z_val = float(loc.attrib.get("z", "0"))
            except Exception:
                continue
            # Prefer population id as neuron name
            if pop_id:
                positions[pop_id] = (x_val, y_val, z_val)
                positions[norm_name(pop_id)] = (x_val, y_val, z_val)
            # Fallback: instance id if it matches a neuron name
            inst_id = inst.attrib.get("id", "")
            if inst_id:
                positions[inst_id] = (x_val, y_val, z_val)
                positions[norm_name(inst_id)] = (x_val, y_val, z_val)

    # Keep only neurons in the connectome intersection, preserve order
    pos_arr = np.zeros((len(neurons), 3), dtype=float)
    missing = []
    for i, n in enumerate(neurons):
        n_norm = norm_name(n)
        if n in positions:
            pos_arr[i] = positions[n]
        elif n_norm in positions:
            pos_arr[i] = positions[n_norm]
        else:
            missing.append(n)
    if missing:
        raise ValueError(f"NeuroML positions missing for {len(missing)} neurons (e.g., {missing[:5]}).")

    return x, pos_arr


def _load_stimuli():
    """Load optional stimulus events for future interactive use.

    Events schema (list of dicts):
      {"type": "touch|temperature|odor|light|food",
       "t_start": float, "t_end": float,
       "strength": float,
       "target": "head|tail|body|global",
       "segment": int (optional)}
    """
    payload = os.environ.get("CE_STIMULI_JSON", "").strip()
    path = os.environ.get("CE_STIMULI_PATH", "").strip()
    if path:
        try:
            payload = Path(path).read_text()
        except Exception:
            payload = ""
    if not payload:
        return []
    try:
        data = json.loads(payload)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _stimulus_drive(events, t, segs):
    """Return drive and turn bias based on active stimuli (no effect if none)."""
    if not events:
        return 0.0, 0.0, None
    drive = 0.0
    turn_bias = 0.0
    seg_bias = None
    for ev in events:
        try:
            t0 = float(ev.get("t_start", 0.0))
            t1 = float(ev.get("t_end", 0.0))
            if not (t0 <= t <= t1):
                continue
            strength = float(ev.get("strength", 0.0))
            etype = str(ev.get("type", ""))
            target = str(ev.get("target", "global"))
        except Exception:
            continue
        # Map stimulus type to generic drive/turn effects (placeholder hooks)
        if etype in {"temperature", "light", "odor", "food"}:
            drive += 0.1 * strength
            turn_bias += 0.05 * strength
        elif etype == "touch":
            drive -= 0.1 * strength
            turn_bias += 0.15 * strength
        if target == "head":
            seg_bias = 0
        elif target == "tail":
            seg_bias = segs - 1
        elif target == "body":
            seg_bias = segs // 2
        elif isinstance(ev.get("segment"), int):
            seg_bias = int(ev.get("segment"))
    return drive, turn_bias, seg_bias


def _motor_groups(neurons):
    fwd = []
    bwd = []
    for i, n in enumerate(neurons):
        if re.match(r"^(VB|DB)\d+$", n):
            fwd.append(i)
        elif re.match(r"^(VA|DA)\d+$", n):
            bwd.append(i)
    return fwd, bwd


def simulate_c_elegans():
    seed = int(os.environ.get("CE_SEED", "42"))
    rng = np.random.default_rng(seed)

    xlsx_path, neurons, chem_nn, gap_nn = load_connectome_neuron_submatrices()
    neuroml_path, neuron_pos = _load_neuroml_positions(neurons)
    n = len(neurons)

    # ==========================================================================
    # CONNECTIVITY MATRICES
    # ==========================================================================
    
    # Normalize chemical weights for stability
    out_strength = np.maximum(1.0, chem_nn.sum(axis=1, keepdims=True))
    W = chem_nn / out_strength

    # Gap coupling Laplacian (distance-weighted by spatial positions)
    if neuron_pos is not None and neuron_pos.shape[0] == n:
        diff = neuron_pos[:, None, :] - neuron_pos[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        scale = np.median(dist[dist > 0]) if np.any(dist > 0) else 1.0
        scale = max(scale, 1.0)
        dist_w = np.exp(-dist / scale)
        gap_eff = gap_nn * dist_w
    else:
        gap_eff = gap_nn
    G = gap_eff / np.maximum(1.0, gap_eff.sum(axis=1, keepdims=True))
    deg = np.sum(G, axis=1)
    L = np.diag(deg) - G

    # ==========================================================================
    # NEURON IDENTIFICATION
    # ==========================================================================
    
    # Command interneurons
    avb_idx = [i for i, name in enumerate(neurons) if name.startswith('AVB')]
    ava_idx = [i for i, name in enumerate(neurons) if name.startswith('AVA')]
    
    # Motor neurons
    fwd_idx, bwd_idx = _motor_groups(neurons)
    if len(fwd_idx) == 0:
        fwd_idx = list(range(0, min(10, n)))
    if len(bwd_idx) == 0:
        bwd_idx = list(range(min(10, n), min(20, n)))
    
    # Ventral/dorsal motor neurons
    ventral_idx = [i for i, name in enumerate(neurons) if re.match(r'^(VB|VA)\d+$', name)]
    dorsal_idx = [i for i, name in enumerate(neurons) if re.match(r'^(DB|DA)\d+$', name)]
    if len(ventral_idx) == 0:
        ventral_idx = list(range(0, min(5, n)))
    if len(dorsal_idx) == 0:
        dorsal_idx = list(range(min(5, n), min(10, n)))
    
    # Sensory neurons (for triggering behaviors)
    touch_neurons = [i for i, name in enumerate(neurons) if name.startswith(('ALM', 'AVM', 'PLM', 'PVM'))]
    
    # ==========================================================================
    # NEURAL STATE VARIABLES
    # ==========================================================================
    
    V = rng.random(n) * 0.1          # Membrane potential
    a = np.zeros(n)                   # Firing rate
    w = np.zeros(n)                   # Adaptation current
    D = np.ones(n)                    # Synaptic depression
    F = np.ones(n)                    # Synaptic facilitation
    
    # ==========================================================================
    # PARAMETERS
    # ==========================================================================
    
    # Neural dynamics
    tau_m = 0.12                      # Membrane time constant
    V_rest = 0.0
    V_thresh = 1.0
    noise_std = 0.08
    
    # Adaptation
    tau_w = 0.25
    adapt_strength = 0.15
    
    # Synaptic plasticity
    U_depression = 0.5
    tau_rec = 0.8
    tau_facil = 0.35
    U_facil = 0.18
    
    # Connection strengths
    beta_chem = 0.85
    gamma_gap = 0.06
    bias_current = 0.8

    # Glia / hypodermis / plasticity proxies
    tau_glia = 1.5
    glia_gain = 0.4
    tau_hypo = 2.0
    hypo_gain = 0.35
    tau_plastic = 6.0
    target_activity = 0.35
    
    # Reciprocal inhibition between AVA and AVB (creates behavioral switching)
    ava_avb_inhibition = 0.5
    
    # ==========================================================================
    # BODY / KINEMATICS PARAMETERS
    # ==========================================================================
    
    steps = 600
    dt = 0.05
    segs = 10
    
    # Oscillator parameters (CPG)
    omega_forward = 0.45 * 2 * np.pi   # Forward crawl frequency ~0.45 Hz
    omega_reverse = 0.55 * 2 * np.pi   # Reversal frequency (slightly faster)
    wavelength = 1.5                    # Body wavelengths
    
    # Curvature parameters
    curv_amp_base = 0.6                # Base curvature amplitude
    curv_amp_head = 1.3                # Head bends more
    curv_amp_tail = 0.5                # Tail bends less
    
    # Movement parameters
    base_speed = 0.18
    speed_gain = 0.12
    turn_gain = 0.4
    
    # Omega turn parameters
    omega_turn_curvature = 0.9         # Deep bend during omega turn
    omega_turn_rate = 2.5              # Rotation speed during omega turn
    
    # ==========================================================================
    # BEHAVIORAL STATE MACHINE
    # ==========================================================================
    
    # States: 'forward', 'reversal', 'omega_turn', 'pause'
    state = 'forward'
    state_timer = 0.0
    
    # State durations (seconds)
    min_forward_duration = 2.0
    min_reversal_duration = 0.8
    max_reversal_duration = 2.0
    omega_turn_duration = 1.2
    pause_duration = 0.3
    
    # Transition thresholds
    reversal_threshold = 0.65          # AVA activity to trigger reversal
    forward_threshold = 0.60           # AVB activity to resume forward
    spontaneous_reversal_rate = 0.02   # Random reversals per second
    
    # ==========================================================================
    # STATE VARIABLES
    # ==========================================================================
    
    theta = 0.0                        # Heading angle
    pos = np.zeros(2, dtype=float)     # Position
    phase = np.linspace(0, -2*np.pi * wavelength, segs)  # Phase along body (traveling wave)
    
    prev_curv = np.zeros(segs)
    muscle_state = np.zeros(segs)      # Muscle activation state
    omega_turn_dir = 1.0               # Turn direction (set once per omega turn)

    # Slow modulators (proxies)
    glia_state = 0.5
    hypodermis_state = 0.5
    syn_gain = 1.0
    
    # Segment amplitude gradient (head > tail)
    segment_amps = np.linspace(curv_amp_head, curv_amp_tail, segs)

    # Optional stimuli (hooks only; normal runs have no events)
    stim_events = _load_stimuli()

    # Map motor neurons to body segments using spatial positions (AP axis)
    motor_seg_idx = None
    if neuron_pos is not None and neuron_pos.shape[0] == n:
        y_vals = neuron_pos[:, 1]
        y_min = float(np.min(y_vals))
        y_max = float(np.max(y_vals))
        denom = max(y_max - y_min, 1.0)
        motor_seg_idx = np.full(n, -1, dtype=int)
        motor_indices = set(fwd_idx + bwd_idx + ventral_idx + dorsal_idx)
        for i in motor_indices:
            frac = (y_vals[i] - y_min) / denom
            seg = int(np.clip(round(frac * (segs - 1)), 0, segs - 1))
            motor_seg_idx[i] = seg
    
    # ==========================================================================
    # OUTPUT ARRAYS
    # ==========================================================================
    
    n_out = min(n, 100)  # Output up to 100 neurons
    positions = np.zeros((steps, 2), dtype=float)
    velocities = np.zeros((steps, 2), dtype=float)
    curvature = np.zeros((steps, segs), dtype=float)
    neural_out = np.zeros((steps, n_out), dtype=float)
    
    # Track behavioral states for analysis
    state_history = []
    
    # ==========================================================================
    # MAIN SIMULATION LOOP
    # ==========================================================================
    
    for t in range(steps):
        time = t * dt
        state_timer += dt

        # ======================================================================
        # 0. SLOW MODULATORS (GLIA / HYPODERMIS PROXIES)
        # ======================================================================

        mean_a = float(np.mean(a)) if n > 0 else 0.0
        glia_state += (mean_a - glia_state) * dt / tau_glia
        glia_state = float(np.clip(glia_state, 0.0, 1.0))

        mean_curv = float(np.mean(np.abs(prev_curv))) if prev_curv.size > 0 else 0.0
        hypodermis_state += (mean_curv - hypodermis_state) * dt / tau_hypo
        hypodermis_state = float(np.clip(hypodermis_state, 0.0, 1.0))
        
        # ======================================================================
        # 1. SYNAPTIC PLASTICITY UPDATE
        # ======================================================================
        
        D += (1.0 - D) / tau_rec * dt
        F += (1.0 - F) / tau_facil * dt + U_facil * a * dt
        F = np.clip(F, 1.0, 2.5)
        
        # ======================================================================
        # 2. COMPUTE SYNAPTIC CURRENTS
        # ======================================================================
        
        chem_current = W.T @ (a * D * F)
        gap_current = -L @ V

        beta_chem_eff = beta_chem * np.clip(syn_gain, 0.7, 1.3)
        gamma_gap_eff = gamma_gap * (1.0 + glia_gain * (glia_state - 0.5))
        gamma_gap_eff = float(np.clip(gamma_gap_eff, 0.02, 0.12))

        I_syn = beta_chem_eff * chem_current + gamma_gap_eff * gap_current + bias_current - w
        
        # ======================================================================
        # 3. RECIPROCAL INHIBITION (AVA <-> AVB)
        # ======================================================================
        
        if len(avb_idx) > 0 and len(ava_idx) > 0:
            ava_activity = np.mean(a[ava_idx])
            avb_activity = np.mean(a[avb_idx])
            
            # Mutual inhibition creates bistable switching
            I_syn[avb_idx] -= ava_avb_inhibition * ava_activity
            I_syn[ava_idx] -= ava_avb_inhibition * avb_activity
            
            # State-dependent bias
            if state == 'forward':
                I_syn[avb_idx] += 0.2  # Bias toward forward
            elif state == 'reversal':
                I_syn[ava_idx] += 0.2  # Bias toward reversal
        else:
            ava_activity = 0.5
            avb_activity = 0.5
        
        # ======================================================================
        # 4. ADD NOISE
        # ======================================================================
        
        noise_scale = noise_std * (1.0 + 0.3 * (glia_state - 0.5))
        I_syn += rng.normal(0, noise_scale, n)
        
        # ======================================================================
        # 5. NEURAL DYNAMICS (LIF)
        # ======================================================================
        
        dV = (-(V - V_rest) + I_syn) / tau_m * dt
        V += dV
        
        # Firing rate (soft threshold) with bounded input
        z = 8.0 * (V - V_thresh)
        z = np.clip(z, -10.0, 10.0)
        a = 1.0 / (1.0 + np.exp(-z))

        # Homeostatic plasticity (slow gain adaptation)
        syn_gain += (target_activity - float(np.mean(a))) * dt / tau_plastic
        syn_gain = float(np.clip(syn_gain, 0.7, 1.3))
        
        # Update synaptic depression
        D -= U_depression * a * dt
        D = np.clip(D, 0.1, 1.0)
        
        # Update adaptation
        w += (-w / tau_w + adapt_strength * a) * dt
        
        # ======================================================================
        # 6. BEHAVIORAL STATE MACHINE
        # ======================================================================
        
        prev_state = state
        
        if state == 'forward':
            # Check for reversal trigger
            if state_timer > min_forward_duration:
                # Neural trigger
                if ava_activity > reversal_threshold:
                    state = 'reversal'
                    state_timer = 0.0
                # Spontaneous reversal
                elif rng.random() < spontaneous_reversal_rate * dt:
                    state = 'reversal'
                    state_timer = 0.0
                    
        elif state == 'reversal':
            # Check for omega turn or return to forward
            if state_timer > max_reversal_duration:
                # 70% chance of omega turn after reversal
                if rng.random() < 0.7:
                    state = 'omega_turn'
                    omega_turn_dir = 1.0 if rng.random() > 0.5 else -1.0
                else:
                    state = 'forward'
                state_timer = 0.0
            elif state_timer > min_reversal_duration and avb_activity > forward_threshold:
                state = 'forward'
                state_timer = 0.0
                
        elif state == 'omega_turn':
            if state_timer > omega_turn_duration:
                state = 'forward'
                state_timer = 0.0
                
        elif state == 'pause':
            if state_timer > pause_duration:
                state = 'forward'
                state_timer = 0.0
        
        state_history.append(state)
        
        # ======================================================================
        # 7. OSCILLATOR / WAVE GENERATION
        # ======================================================================
        
        stim_drive, stim_turn, stim_seg = _stimulus_drive(stim_events, time, segs)

        if state == 'forward':
            # Forward traveling wave (head to tail)
            omega = omega_forward * (0.9 + 0.2 * avb_activity)  # Modulated by AVB
            phase += omega * dt
            wave_direction = 1.0
            amplitude_mod = 1.0
            
        elif state == 'reversal':
            # Reverse traveling wave (tail to head)
            omega = omega_reverse * (0.9 + 0.2 * ava_activity)  # Modulated by AVA
            phase -= omega * dt  # Reverse direction
            wave_direction = -1.0
            amplitude_mod = 0.8  # Slightly reduced amplitude
            
        elif state == 'omega_turn':
            # Deep ventral bend at head, suppressed tail
            omega = omega_forward * 0.5  # Slow oscillation
            phase += omega * dt
            wave_direction = 1.0
            amplitude_mod = 0.6
            
        else:  # pause
            omega = 0
            amplitude_mod = 0.2
        
        # ======================================================================
        # 8. CURVATURE COMPUTATION
        # ======================================================================
        
        # Base traveling wave
        phase_along_body = phase[0] - np.linspace(0, 2*np.pi * wavelength, segs) * wave_direction
        target_curv = curv_amp_base * amplitude_mod * segment_amps * np.sin(phase_along_body)

        # Spatially informed motor drive modulation (segment-specific)
        if motor_seg_idx is not None:
            seg_drive = np.zeros(segs, dtype=float)
            seg_count = np.zeros(segs, dtype=float)
            active = np.where(motor_seg_idx >= 0)[0]
            if active.size > 0:
                for i in active:
                    sidx = motor_seg_idx[i]
                    seg_drive[sidx] += a[i]
                    seg_count[sidx] += 1.0
                seg_count = np.maximum(seg_count, 1.0)
                seg_drive = seg_drive / seg_count
                seg_drive = (seg_drive - seg_drive.mean()) if segs > 1 else seg_drive
                target_curv = target_curv * (1.0 + 0.2 * seg_drive)

        # Apply stimulus segment bias (future interactive hook)
        if stim_seg is not None and 0 <= stim_seg < segs:
            target_curv[stim_seg] += 0.2 * stim_drive
        
        # Special handling for omega turn
        if state == 'omega_turn':
            # Deep ventral bend in head segments
            omega_progress = min(state_timer / omega_turn_duration, 1.0)
            head_bend = omega_turn_curvature * np.sin(np.pi * omega_progress)
            target_curv[0:3] = -head_bend * np.array([1.0, 0.8, 0.5])  # Ventral = negative
            target_curv[3:] *= 0.3  # Suppress tail movement
        
        # Muscle dynamics (low-pass filter) modulated by hypodermis proxy
        muscle_tau = 0.15 * (1.0 + 0.6 * (hypodermis_state - 0.5))
        muscle_tau = float(np.clip(muscle_tau, 0.08, 0.25))
        muscle_state += (target_curv - muscle_state) * dt / muscle_tau
        
        # Final curvature (with proprioceptive coupling between segments)
        coupling_strength = 0.15 * (1.0 + hypo_gain * (hypodermis_state - 0.5))
        coupling_strength = float(np.clip(coupling_strength, 0.05, 0.30))
        curv = muscle_state.copy()
        for i in range(1, segs - 1):
            curv[i] += coupling_strength * (muscle_state[i-1] - 2*muscle_state[i] + muscle_state[i+1])
        
        curv = np.clip(curv, -1.0, 1.0)
        curvature[t] = curv
        prev_curv = curv.copy()
        
        # ======================================================================
        # 9. LOCOMOTION
        # ======================================================================
        
        if state == 'forward':
            # Forward speed based on AVB activity
            drive = 0.5 + 0.5 * avb_activity + stim_drive
            speed = base_speed + speed_gain * drive
            
            # Gradual turning from body curvature
            turn_moment = np.sum(curv * np.linspace(-1, 1, segs))
            theta += turn_gain * (turn_moment + stim_turn) * dt
            
        elif state == 'reversal':
            # Backward speed based on AVA activity
            drive = 0.5 + 0.5 * ava_activity + stim_drive
            speed = -(base_speed * 0.7 + speed_gain * 0.5 * drive)  # Slower backward
            
            # Turning is reduced during reversal
            turn_moment = np.sum(curv * np.linspace(-1, 1, segs))
            theta += turn_gain * 0.3 * (turn_moment + stim_turn) * dt
            
        elif state == 'omega_turn':
            # Minimal forward/backward movement
            speed = base_speed * 0.1

            # Consistent rotation for entire turn (direction set on state entry)
            theta += omega_turn_dir * omega_turn_rate * dt
            
        else:  # pause
            speed = 0.0
        
        # Velocity and position update
        v = speed * np.array([np.cos(theta), np.sin(theta)])
        pos = pos + v * dt
        
        positions[t] = pos
        velocities[t] = v
        
        # ======================================================================
        # 10. RECORD NEURAL OUTPUT
        # ======================================================================
        
        neural_out[t] = a[:n_out]
    
    # ==========================================================================
    # RETURN RESULTS
    # ==========================================================================
    
    # Count behavioral states
    state_counts = {s: state_history.count(s) for s in ['forward', 'reversal', 'omega_turn', 'pause']}
    
    # Connectome stats
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
        "n_output_neurons": int(n_out),
        "chem_sum": chem_sum,
        "chem_nnz": chem_nnz,
        "gap_sum": gap_sum,
        "gap_nnz": gap_nnz,
        "source_xlsx": Path(xlsx_path).name,
        "source_neuroml": Path(neuroml_path).name,
        "behavioral_states": state_counts,
    }
# EVOLVE-BLOCK-END


if __name__ == "__main__":
    out = simulate_c_elegans()
    print(json.dumps(out))
