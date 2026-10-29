#!/usr/bin/env python3
"""
CREATE XLSX FILE WITH make_matrix_excel.py!!!!
same thing with "sim_2d_no_geology_5_random_shots.py" but with geology on the background which is matrix btw.
format: 1080x1080, 60 FPS, 4.5 sec
37x37 particle grid, red star source (6.0, 6.0)
background acoustic wave (red compression, blue dilatation) + Viridis vel
"""

import os
import pandas as pd
import numpy as np
import torch
import deepwave
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import RegularGridInterpolator

def main():
    excel_path = './vel_matrix.xlsx' # create xlsx with make_matrix_excel.py btw
    output_mp4 = './wave_propagation_2d_with_user_geology.mp4'
    
    print(f"read excel: {excel_path}...")
    df = pd.read_excel(excel_path, sheet_name='vel_matrix_37x37', index_col=0)

    x_balls = np.array(df.columns, dtype=float)
    z_balls_desc = np.array(df.index, dtype=float)
    z_balls = z_balls_desc[::-1]
    vel_matrix = df.values[::-1, :]

    L = 12.0
    nx, nz = 400, 400
    dx = L / nx # 0.03 m
    x_sim = np.linspace(0.5 * dx, L - 0.5 * dx, nx)
    z_sim = np.linspace(0.5 * dx, L - 0.5 * dx, nz)
    Z_mesh, X_mesh = np.meshgrid(z_sim, x_sim, indexing='ij')

    # interpolation
    Z_clamped = np.clip(Z_mesh, z_balls[0], z_balls[-1])
    X_clamped = np.clip(X_mesh, x_balls[0], x_balls[-1])
    interp_linear = RegularGridInterpolator((z_balls, x_balls), vel_matrix, method='linear')
    v_sim = interp_linear((Z_clamped, X_clamped))

    # grid (37x37)
    X_b, Z_b = np.meshgrid(x_balls, z_balls)
    X_eq = X_b.flatten()
    Z_eq = Z_b.flatten()

    x0, z0 = 6.0, 6.0
    dX = X_eq - x0
    dZ = Z_eq - z0
    R_eq = np.sqrt(dX**2 + dZ**2)
    R_safe = np.where(R_eq == 0, 1.0, R_eq)
    cos_theta = dX / R_safe
    sin_theta = dZ / R_safe

    # FDTD Parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    v_torch = torch.tensor(v_sim, dtype=torch.float32, device=device)

    duration = 4.5 # sec
    fps = 60
    total_frames = int(fps * duration) # 270 frame
    dt_sim = 0.002 # due to CFL (v_max = 7.0 m/s)
    nt_sim = int(duration / dt_sim) # 2250 step
    freq = 1.5 # Hz

    source_amplitudes = deepwave.wavelets.ricker(freq, nt_sim, dt_sim, 1.0 / freq).reshape(1, 1, -1).to(device)
    source_locations = torch.zeros(1, 1, 2, dtype=torch.long, device=device)
    source_locations[0, 0, 0] = nz // 2 # 200 (z = 6.0)
    source_locations[0, 0, 1] = nx // 2 # 200 (x = 6.0)

    step_interval = nt_sim // total_frames # her video karesi için FDTD adımı

    p_frames = []
    def forward_cb(state):
        wf = state.get_wavefield('wavefield_0')
        p_frames.append(wf.detach().clone())

    print(f"Deepwave FDTD for ({nz}x{nx}, {nt_sim} step, max_v = {v_sim.max():.2f} km/s) on {device}...")
    deepwave.scalar(
        v_torch, dx, dt_sim,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        forward_callback=forward_cb,
        callback_frequency=step_interval,
        pml_freq=freq,
        pml_width=30
    )

    p_tensor = torch.stack(p_frames[:total_frames], dim=0).squeeze()
    ix_part = np.clip(np.round(X_eq / dx).astype(int), 0, nx - 1)
    iz_part = np.clip(np.round(Z_eq / dx).astype(int), 0, nz - 1)
    dt_frame = 1.0 / fps
    # print("particle wont fucking work amına koyim")
    p_particles = p_tensor[:, iz_part, ix_part].cpu().numpy()
    u_r = np.cumsum(p_particles, axis=0) * dt_frame
    gain_part = np.sqrt(np.maximum(R_eq, 0.4) / 0.8)[np.newaxis, :]
    u_r_comp = u_r * gain_part
    u_r_comp[:, R_eq < 0.2] = 0.0
    # print("dont know why")
    mask_mid = (R_eq >= 1.0) & (R_eq <= 4.5)
    peak_val = np.percentile(np.abs(u_r_comp[:, mask_mid]), 99.5)
    target_disp = 0.12
    disp_scale = target_disp / peak_val if peak_val > 0 else 1.0
    u_r_scaled = u_r_comp * disp_scale
    zz, xx = np.meshgrid(np.arange(nz), np.arange(nx), indexing='ij')
    r_field = np.sqrt((xx - nx // 2)**2 + (zz - nz // 2)**2) * dx
    gain_field = np.sqrt(np.maximum(r_field, 0.4) / 0.8)
    cmap_wave = LinearSegmentedColormap.from_list(
        'clean_bwr', 
        [
            (0.00, '#2563EB'), # Mavi (Genleşme)
            (0.25, '#93C5FD'),
            (0.50, '#FFFFFF'), # Saf Beyaz (Durgun)
            (0.75, '#FCA5A5'),
            (1.00, '#DC2626')  # Kırmızı (Sıkışma)
        ]
    )

    is_star = (np.abs(X_eq - x0) < 0.15) & (np.abs(Z_eq - z0) < 0.15)
    other_mask = ~is_star
    fig = plt.figure(figsize=(9.0, 9.0), dpi=120, facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1], facecolor='white')
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect('equal')
    ax.axis('off')
    # Viridis (geo of user)
    ax.imshow(v_sim, extent=[0, L, 0, L], origin='lower', cmap='viridis', alpha=0.28, zorder=0)
    grid_steps = np.arange(0, 12.1, 1.2)
    for gv in grid_steps:
        is_c = abs(gv - 6.0) < 1e-3
        ax.axhline(gv, color='#94A3B8' if is_c else '#CBD5E1', lw=1.2 if is_c else 0.6, alpha=0.4, zorder=1)
        ax.axvline(gv, color='#94A3B8' if is_c else '#CBD5E1', lw=1.2 if is_c else 0.6, alpha=0.4, zorder=1)

    # Propagate!
    vmax_val = 0.0001
    init_snap = p_tensor[0].cpu().numpy() * gain_field
    im_wave = ax.imshow(
        init_snap,
        extent=[0, L, 0, L],
        origin='lower',
        cmap=cmap_wave,
        vmin=-vmax_val,
        vmax=vmax_val,
        alpha=0.68,
        zorder=2
    )

    # particles
    balls_other = ax.scatter(
        X_eq[other_mask],
        Z_eq[other_mask],
        s=24,
        c='#0F172A',
        edgecolors='none',
        alpha=0.92,
        zorder=5
    )

    # source
    ax.scatter([x0], [z0], s=280, marker='*', c='#DC2626', edgecolors='white', linewidth=1.2, zorder=10)

    print("frame time...")
    def update(frame):
        snap = p_tensor[frame].cpu().numpy() * gain_field
        im_wave.set_data(snap)

        ur_now = u_r_scaled[frame]
        X_cur = X_eq + ur_now * cos_theta
        Z_cur = Z_eq + ur_now * sin_theta

        coords = np.column_stack((X_cur[other_mask], Z_cur[other_mask]))
        balls_other.set_offsets(coords)

        return im_wave, balls_other

    anim = animation.FuncAnimation(fig, update, frames=total_frames, interval=1000/fps, blit=False)
    writer = animation.FFMpegWriter(fps=fps, codec='libx264', extra_args=['-pix_fmt', 'yuv420p', '-crf', '18'])
    anim.save(output_mp4, writer=writer)
    plt.close(fig)
    print(f"Video done: {output_mp4}")

if __name__ == '__main__':
    main()
