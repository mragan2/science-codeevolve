#!/usr/bin/env python3
"""
C. elegans Locomotion Simulator - Visualization Tool

Usage:
    python best_sol.py | python visualize_elegans.py
    # OR
    python best_sol.py > output.json && python visualize_elegans.py output.json
    # OR
    python visualize_elegans.py --demo  # Run with synthetic data
"""

import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec

# ============================================================================
# Data Loading
# ============================================================================

def load_data(source=None):
    """Load simulation data from file, stdin, or generate demo data."""
    if source == '--demo':
        return generate_demo_data()
    
    if source and source != '-':
        with open(source, 'r') as f:
            data = json.load(f)
    else:
        # Read from stdin
        raw = sys.stdin.read().strip()
        # Handle multiple JSON lines (take last valid one)
        for line in reversed(raw.split('\n')):
            try:
                data = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        else:
            raise ValueError("No valid JSON found in input")
    
    return data


def generate_demo_data():
    """Generate synthetic demo data for testing visualization."""
    np.random.seed(42)
    steps = 400
    segs = 10
    dt = 0.05
    
    # Generate sinusoidal traveling wave for curvature
    t = np.arange(steps) * dt
    phase_offsets = np.linspace(0, 2*np.pi, segs)
    omega = 0.4 * 2 * np.pi
    
    curvature = np.zeros((steps, segs))
    for i in range(segs):
        amp = np.exp(-i * 0.3)  # Head-biased amplitude
        curvature[:, i] = amp * np.sin(omega * t - phase_offsets[i])
    
    # Generate trajectory
    positions = np.zeros((steps, 2))
    theta = 0
    pos = np.array([0.0, 0.0])
    for i in range(steps):
        speed = 0.15 + 0.05 * np.sin(omega * t[i])
        theta += 0.3 * np.mean(curvature[i]) * dt
        vel = speed * np.array([np.cos(theta), np.sin(theta)])
        pos = pos + vel * dt
        positions[i] = pos
    
    # Generate neural activity
    neural = np.zeros((steps, 32))
    for i in range(32):
        freq = 0.2 + 0.1 * (i / 32)
        neural[:, i] = 0.5 + 0.3 * np.sin(2 * np.pi * freq * t + i * 0.3)
        neural[:, i] += 0.1 * np.random.randn(steps)
    neural = np.clip(neural, 0, 1)
    
    return {
        'positions': positions.tolist(),
        'curvature': curvature.tolist(),
        'neural': neural.tolist(),
        'dt': dt,
        'n_neurons': 32,
        'velocities': (np.diff(positions, axis=0, prepend=positions[:1]) / dt).tolist()
    }


# ============================================================================
# Static Plots
# ============================================================================

def plot_summary(data, save_path=None):
    """Create a 2x2 summary figure with key visualizations."""
    positions = np.array(data['positions'])
    curvature = np.array(data['curvature'])
    neural = np.array(data['neural'])
    dt = data['dt']
    
    steps, segs = curvature.shape
    time = np.arange(steps) * dt
    
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # ---- 1. Trajectory (top-left) ----
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Color trajectory by time
    points = positions.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(0, steps * dt)
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(time[:-1])
    lc.set_linewidth(2)
    ax1.add_collection(lc)
    
    ax1.plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start')
    ax1.plot(positions[-1, 0], positions[-1, 1], 'r*', markersize=15, label='End')
    ax1.set_xlim(positions[:, 0].min() - 0.5, positions[:, 0].max() + 0.5)
    ax1.set_ylim(positions[:, 1].min() - 0.5, positions[:, 1].max() + 0.5)
    ax1.set_xlabel('X position')
    ax1.set_ylabel('Y position')
    ax1.set_title('Worm Trajectory (colored by time)')
    ax1.legend()
    ax1.set_aspect('equal')
    cbar1 = plt.colorbar(lc, ax=ax1)
    cbar1.set_label('Time (s)')
    
    # ---- 2. Curvature Heatmap (top-right) ----
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(curvature.T, aspect='auto', cmap='RdBu_r',
                     extent=[0, steps*dt, segs-0.5, -0.5],
                     vmin=-1, vmax=1)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Body segment (0=head)')
    ax2.set_title('Curvature Kymograph')
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Curvature')
    
    # ---- 3. Neural Activity Heatmap (bottom-left) ----
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(neural.T, aspect='auto', cmap='hot',
                     extent=[0, steps*dt, neural.shape[1]-0.5, -0.5],
                     vmin=0, vmax=1)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Neuron index')
    ax3.set_title('Neural Activity')
    cbar3 = plt.colorbar(im3, ax=ax3)
    cbar3.set_label('Firing rate')
    
    # ---- 4. Time Series (bottom-right) ----
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Mean curvature
    mean_curv = np.mean(np.abs(curvature), axis=1)
    ax4.plot(time, mean_curv, 'b-', label='Mean |curvature|', linewidth=1.5)
    
    # Speed
    if 'velocities' in data:
        velocities = np.array(data['velocities'])
        speed = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2)
        ax4.plot(time, speed, 'g-', label='Speed', linewidth=1.5, alpha=0.7)
    
    # Head curvature
    ax4.plot(time, np.abs(curvature[:, 0]), 'r--', label='Head |curvature|', 
             linewidth=1, alpha=0.7)
    
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Value')
    ax4.set_title('Time Series')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('C. elegans Locomotion Simulation Summary', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved summary plot to: {save_path}")
    
    return fig


def plot_curvature_wave(data, save_path=None):
    """Plot curvature showing traveling wave pattern."""
    curvature = np.array(data['curvature'])
    dt = data['dt']
    steps, segs = curvature.shape
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Top: Curvature traces for each segment
    ax1 = axes[0]
    time = np.arange(steps) * dt
    colors = plt.cm.viridis(np.linspace(0, 1, segs))
    
    for i in range(segs):
        ax1.plot(time, curvature[:, i] + i * 0.5, color=colors[i], 
                linewidth=1.5, label=f'Seg {i}' if i % 3 == 0 else '')
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Curvature (offset for clarity)')
    ax1.set_title('Curvature Waves Along Body (head=segment 0)')
    ax1.legend(loc='upper right')
    ax1.set_xlim(0, min(10, steps * dt))  # Show first 10 seconds
    
    # Bottom: Snapshot of body shape at different times
    ax2 = axes[1]
    snapshot_times = np.linspace(0, steps-1, 8).astype(int)
    
    for idx, t_idx in enumerate(snapshot_times):
        curv = curvature[t_idx]
        # Reconstruct body shape from curvature
        x, y = reconstruct_body_shape(curv)
        # Offset for display
        offset = idx * 1.5
        ax2.plot(x + offset, y, 'b-', linewidth=2)
        ax2.plot(x[0] + offset, y[0], 'ro', markersize=6)  # Head
        ax2.text(offset + 0.5, -0.8, f't={t_idx*dt:.1f}s', ha='center', fontsize=9)
    
    ax2.set_xlabel('X (offset by time)')
    ax2.set_ylabel('Y')
    ax2.set_title('Body Shape Snapshots')
    ax2.set_aspect('equal')
    ax2.set_ylim(-1.5, 1.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved wave plot to: {save_path}")
    
    return fig


def plot_neural_analysis(data, save_path=None):
    """Detailed neural activity analysis."""
    neural = np.array(data['neural'])
    dt = data['dt']
    steps, n_neurons = neural.shape
    time = np.arange(steps) * dt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Mean firing rates per neuron
    ax1 = axes[0, 0]
    mean_rates = np.mean(neural, axis=0)
    std_rates = np.std(neural, axis=0)
    ax1.bar(range(n_neurons), mean_rates, yerr=std_rates, alpha=0.7, capsize=2)
    ax1.set_xlabel('Neuron index')
    ax1.set_ylabel('Mean firing rate')
    ax1.set_title('Firing Rate Statistics per Neuron')
    
    # 2. Correlation matrix
    ax2 = axes[0, 1]
    corr_matrix = np.corrcoef(neural.T)
    im = ax2.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax2.set_xlabel('Neuron')
    ax2.set_ylabel('Neuron')
    ax2.set_title('Neural Correlation Matrix')
    plt.colorbar(im, ax=ax2, label='Correlation')
    
    # 3. Example traces
    ax3 = axes[1, 0]
    sample_neurons = [0, n_neurons//4, n_neurons//2, 3*n_neurons//4, n_neurons-1]
    for i, idx in enumerate(sample_neurons):
        ax3.plot(time, neural[:, idx] + i * 0.3, label=f'N{idx}', linewidth=1)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Firing rate (offset)')
    ax3.set_title('Sample Neuron Traces')
    ax3.legend(loc='upper right')
    ax3.set_xlim(0, min(10, steps * dt))
    
    # 4. Population activity
    ax4 = axes[1, 1]
    pop_mean = np.mean(neural, axis=1)
    pop_std = np.std(neural, axis=1)
    ax4.fill_between(time, pop_mean - pop_std, pop_mean + pop_std, alpha=0.3)
    ax4.plot(time, pop_mean, 'b-', linewidth=2, label='Mean ± std')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Population activity')
    ax4.set_title('Population Firing Rate')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved neural analysis to: {save_path}")
    
    return fig


# ============================================================================
# Animation
# ============================================================================

def reconstruct_body_shape(curvature, seg_length=0.1):
    """Reconstruct 2D body shape from curvature array."""
    n_segs = len(curvature)
    x = np.zeros(n_segs + 1)
    y = np.zeros(n_segs + 1)
    theta = 0
    
    for i in range(n_segs):
        theta += curvature[i] * seg_length
        x[i + 1] = x[i] + seg_length * np.cos(theta)
        y[i + 1] = y[i] + seg_length * np.sin(theta)
    
    return x, y


def animate_worm(data, save_path=None, fps=20):
    """Create animation of worm locomotion."""
    positions = np.array(data['positions'])
    curvature = np.array(data['curvature'])
    dt = data['dt']
    steps, segs = curvature.shape
    
    # Calculate frame skip to achieve target fps
    frame_skip = max(1, int(1 / (fps * dt)))
    frame_indices = range(0, steps, frame_skip)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Global trajectory with current position
    ax1 = axes[0]
    ax1.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.3, linewidth=1)
    trajectory_line, = ax1.plot([], [], 'b-', linewidth=2)
    current_pos, = ax1.plot([], [], 'ro', markersize=10)
    ax1.set_xlim(positions[:, 0].min() - 1, positions[:, 0].max() + 1)
    ax1.set_ylim(positions[:, 1].min() - 1, positions[:, 1].max() + 1)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Trajectory')
    ax1.set_aspect('equal')
    
    # Right: Worm body shape
    ax2 = axes[1]
    worm_line, = ax2.plot([], [], 'b-', linewidth=4)
    head_marker, = ax2.plot([], [], 'ro', markersize=12)
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Body Shape')
    ax2.set_aspect('equal')
    
    time_text = ax2.text(0.02, 0.98, '', transform=ax2.transAxes, 
                         verticalalignment='top', fontsize=12)
    
    def init():
        trajectory_line.set_data([], [])
        current_pos.set_data([], [])
        worm_line.set_data([], [])
        head_marker.set_data([], [])
        time_text.set_text('')
        return trajectory_line, current_pos, worm_line, head_marker, time_text
    
    def animate(frame):
        t_idx = frame_indices[frame]
        
        # Update trajectory
        trajectory_line.set_data(positions[:t_idx+1, 0], positions[:t_idx+1, 1])
        current_pos.set_data([positions[t_idx, 0]], [positions[t_idx, 1]])
        
        # Update worm body
        curv = curvature[t_idx]
        x, y = reconstruct_body_shape(curv)
        worm_line.set_data(x, y)
        head_marker.set_data([x[0]], [y[0]])
        
        time_text.set_text(f't = {t_idx * dt:.2f} s')
        
        return trajectory_line, current_pos, worm_line, head_marker, time_text
    
    anim = FuncAnimation(fig, animate, init_func=init, 
                        frames=len(frame_indices), interval=1000/fps, blit=True)
    
    plt.tight_layout()
    
    if save_path:
        print(f"Saving animation to {save_path}... (this may take a while)")
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=fps)
        else:
            anim.save(save_path, writer='ffmpeg', fps=fps)
        print(f"Saved animation to: {save_path}")
    
    return fig, anim


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Visualize C. elegans simulation')
    parser.add_argument('input', nargs='?', default='-', 
                       help='Input JSON file (default: stdin, use --demo for synthetic data)')
    parser.add_argument('--demo', action='store_true', help='Use synthetic demo data')
    parser.add_argument('--save', type=str, help='Save figures with this prefix')
    parser.add_argument('--animate', action='store_true', help='Show animation')
    parser.add_argument('--save-animation', type=str, help='Save animation to file (.gif or .mp4)')
    parser.add_argument('--no-show', action='store_true', help='Do not display plots')
    
    args = parser.parse_args()
    
    # Load data
    if args.demo:
        print("Using synthetic demo data...")
        data = load_data('--demo')
    else:
        print(f"Loading data from: {args.input if args.input != '-' else 'stdin'}")
        data = load_data(args.input)
    
    # Print summary
    positions = np.array(data['positions'])
    curvature = np.array(data['curvature'])
    neural = np.array(data['neural'])
    
    print(f"\n{'='*50}")
    print("SIMULATION SUMMARY")
    print(f"{'='*50}")
    print(f"Time steps:     {len(positions)}")
    print(f"dt:             {data['dt']} s")
    print(f"Total time:     {len(positions) * data['dt']:.1f} s")
    print(f"Body segments:  {curvature.shape[1]}")
    print(f"Neurons:        {neural.shape[1]}")
    print(f"Trajectory length: {np.sum(np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))):.2f} units")
    print(f"Mean |curvature|:  {np.mean(np.abs(curvature)):.3f}")
    print(f"Mean firing rate:  {np.mean(neural):.3f}")
    print(f"{'='*50}\n")
    
    # Create plots
    save_prefix = args.save or ''
    
    fig1 = plot_summary(data, f"{save_prefix}summary.png" if save_prefix else None)
    fig2 = plot_curvature_wave(data, f"{save_prefix}curvature.png" if save_prefix else None)
    fig3 = plot_neural_analysis(data, f"{save_prefix}neural.png" if save_prefix else None)
    
    if args.animate or args.save_animation:
        fig4, anim = animate_worm(data, args.save_animation)
    
    if not args.no_show:
        print("Displaying plots... (close windows to exit)")
        plt.show()


if __name__ == '__main__':
    main()
