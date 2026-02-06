#!/usr/bin/env python3
"""C. elegans visualization - minimal version"""
import sys, json, numpy as np, matplotlib.pyplot as plt

def load_data():
    raw = sys.stdin.read().strip()
    for line in reversed(raw.split('\n')):
        try: return json.loads(line)
        except: continue
    raise ValueError("No valid JSON")

def main():
    data = load_data()
    pos = np.array(data['positions'])
    curv = np.array(data['curvature'])
    neural = np.array(data['neural'])
    dt = data['dt']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Trajectory
    axes[0,0].plot(pos[:,0], pos[:,1], 'b-', lw=1)
    axes[0,0].plot(pos[0,0], pos[0,1], 'go', ms=10, label='Start')
    axes[0,0].plot(pos[-1,0], pos[-1,1], 'r*', ms=12, label='End')
    axes[0,0].set_title('Trajectory'); axes[0,0].legend(); axes[0,0].axis('equal')
    
    # Curvature kymograph
    im1 = axes[0,1].imshow(curv.T, aspect='auto', cmap='RdBu_r', extent=[0,len(curv)*dt,curv.shape[1],0])
    axes[0,1].set_xlabel('Time (s)'); axes[0,1].set_ylabel('Segment'); axes[0,1].set_title('Curvature')
    plt.colorbar(im1, ax=axes[0,1])
    
    # Neural activity
    im2 = axes[1,0].imshow(neural.T, aspect='auto', cmap='hot', extent=[0,len(neural)*dt,neural.shape[1],0])
    axes[1,0].set_xlabel('Time (s)'); axes[1,0].set_ylabel('Neuron'); axes[1,0].set_title('Neural Activity')
    plt.colorbar(im2, ax=axes[1,0])
    
    # Time series
    t = np.arange(len(curv)) * dt
    axes[1,1].plot(t, np.mean(np.abs(curv), axis=1), 'b-', label='Mean |curv|')
    axes[1,1].plot(t, np.abs(curv[:,0]), 'r--', alpha=0.7, label='Head |curv|')
    axes[1,1].set_xlabel('Time (s)'); axes[1,1].legend(); axes[1,1].set_title('Curvature Time Series')
    
    plt.tight_layout()
    plt.savefig('simulation_plot.png', dpi=150)
    print("Saved: simulation_plot.png")
    plt.show()

if __name__ == '__main__': main()
