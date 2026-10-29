#!/usr/bin/env python3
"""
this code creates a 2D acoustic wave propagation simulation.
5 random (uncorrelated with each other) points. cuda support available.
particules shown as points. wave shown with red, blue, white colors.
- Source: Red Star (marker='*')
- Resolution: 1080x1080 square video (figsize=(9.0, 9.0), dpi=120)
- Background: Pure White (#FFFFFF), no writing or whatsoever.
- Outputs:
  1. 1.mp4 (Left-Down: 3.30, 3.60)
  2. 2.mp4 (Right-Up: 8.70, 8.10)
  3. 3.mp4 (Left-Up: 3.60, 8.40)
  4. 4.mp4 (Right-Down: 8.40, 3.30)
  5. 5.mp4 (Close-center: 5.70, 6.30)
"""

import os
import sys
import numpy as np
import torch
import deepwave
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

def main():
    fps = 60
    duration = 5.0 # in second
    total_frames = int(fps * duration) # 300
    fig_size = 9.0
    dpi = 120 # 1080x1080 px
    L = 12.0

    # Deepwave simulation (CUDA)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nz, nx = 400, 400
    dx = L / nx # 0.03 m
    c_vel = 2.4 # m/s
    v = c_vel * torch.ones(nz, nx, device=device)

    dt_sim = 0.004
    nt_sim = int(duration / dt_sim) # 1250 FDTD step
    freq = 1.2 # Hz
    step_interval = nt_sim // total_frames # 4

    source_wavelet = deepwave.wavelets.ricker(freq, nt_sim, dt_sim, 1.0 / freq).reshape(1, 1, -1).to(device)

    # 2D Parçacık Izgarası (37 x 37 = 1369 Parçacık)
    d_ball = 0.30
    x_vals = np.arange(0.6, 11.41, d_ball)
    z_vals = np.arange(0.6, 11.41, d_ball)
    X_eq_grid, Z_eq_grid = np.meshgrid(x_vals, z_vals)
    X_eq = X_eq_grid.flatten()
    Z_eq = Z_eq_grid.flatten()

    ix_part = np.clip(np.round(X_eq / dx).astype(int), 0, nx - 1)
    iz_part = np.clip(np.round(Z_eq / dx).astype(int), 0, nz - 1)
    dt_frame = 1.0 / fps

    # 5 Point
    shot_coords = [
        (3.30, 3.60),
        (8.70, 8.10),
        (3.60, 8.40),
        (8.40, 3.30),
        (5.70, 6.30)
    ]

    cmap_wave = LinearSegmentedColormap.from_list(
        'clean_bwr', 
        [
            (0.00, '#2563EB'), # Blue (-1, expansion)
            (0.25, '#93C5FD'),
            (0.50, '#FFFFFF'), # White (0, still)
            (0.75, '#FCA5A5'),
            (1.00, '#DC2626')  # Kırmızı (+1, compression)
        ]
    )

    out_dir = './'
    grid_steps = np.arange(0, 12.1, 1.2)

    for idx, (x_src, z_src) in enumerate(shot_coords, 1):
        output_mp4 = os.path.join(out_dir, f'wave_2d_shot_{idx}.mp4')
        print(f"\n==================================================")
        print(f"[{idx}/5] Shot: Source loc = ({x_src:.2f}, {z_src:.2f}) -> {output_mp4}")
        print(f"==================================================")

        iz_src = int(round(z_src / dx))
        ix_src = int(round(x_src / dx))
        source_locations = torch.zeros(1, 1, 2, dtype=torch.long, device=device)
        source_locations[0, 0, 0] = iz_src
        source_locations[0, 0, 1] = ix_src

        p_frames = []
        def forward_cb(state):
            wf = state.get_wavefield('wavefield_0')
            p_frames.append(wf.detach().clone())

        deepwave.scalar(
            v, dx, dt_sim,
            source_amplitudes=source_wavelet,
            source_locations=source_locations,
            forward_callback=forward_cb,
            callback_frequency=step_interval,
            pml_freq=freq,
            pml_width=30
        )

        p_tensor = torch.stack(p_frames[:total_frames], dim=0).squeeze()

        # radial geometry and displacements
        dX = X_eq - x_src
        dZ = Z_eq - z_src
        R_eq = np.sqrt(dX**2 + dZ**2)
        R_safe = np.where(R_eq == 0, 1.0, R_eq)
        cos_theta = dX / R_safe
        sin_theta = dZ / R_safe

        p_particles = p_tensor[:, iz_part, ix_part].cpu().numpy()
        u_r = np.cumsum(p_particles, axis=0) * dt_frame

        # gain
        gain_part = np.sqrt(np.maximum(R_eq, 0.4) / 0.8)[np.newaxis, :]
        u_r_comp = u_r * gain_part
        u_r_comp[:, R_eq < 0.2] = 0.0

        mask_mid = (R_eq >= 1.0) & (R_eq <= 4.5)
        peak_val = np.percentile(np.abs(u_r_comp[:, mask_mid]), 99.5)
        target_disp = 0.12
        disp_scale = target_disp / peak_val if peak_val > 0 else 1.0
        u_r_scaled = u_r_comp * disp_scale

        # background gain
        zz, xx = np.meshgrid(np.arange(nz), np.arange(nx), indexing='ij')
        r_field = np.sqrt((xx - ix_src)**2 + (zz - iz_src)**2) * dx
        gain_field = np.sqrt(np.maximum(r_field, 0.4) / 0.8)
        vmax_val = 0.00008

        # do not show point in source_locations
        is_star = (np.abs(X_eq - x_src) < 0.15) & (np.abs(Z_eq - z_src) < 0.15)
        other_mask = ~is_star

        # fig plot
        fig = plt.figure(figsize=(fig_size, fig_size), facecolor='white', dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1], facecolor='white')
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_aspect('equal')
        ax.axis('off')

        # background wave field
        init_snap = p_tensor[0].cpu().numpy() * gain_field
        im = ax.imshow(
            init_snap,
            extent=[0, L, 0, L],
            origin='lower',
            aspect='equal',
            cmap=cmap_wave,
            vmin=-vmax_val,
            vmax=vmax_val,
            alpha=0.65,
            zorder=0
        )

        # minimal grid
        for gv in grid_steps:
            is_c = abs(gv - 6.0) < 1e-3
            lw = 1.4 if is_c else 0.7
            col = '#94A3B8' if is_c else '#E2E8F0'
            ax.axhline(gv, color=col, lw=lw, zorder=1)
            ax.axvline(gv, color=col, lw=lw, zorder=1)

        # grid particles
        balls_other = ax.scatter(
            X_eq[other_mask],
            Z_eq[other_mask],
            s=26,
            c='#1E293B',
            edgecolors='none',
            alpha=0.95,
            zorder=5
        )

        # source as red star
        ax.scatter(
            [x_src], [z_src],
            s=260,
            marker='*',
            c='#DC2626',
            edgecolors='#991B1B',
            linewidth=1.2,
            zorder=10
        )

        def update(frame):
            snap = p_tensor[frame].cpu().numpy() * gain_field
            im.set_data(snap)

            ur_now = u_r_scaled[frame]
            X_cur = X_eq + ur_now * cos_theta
            Z_cur = Z_eq + ur_now * sin_theta

            coords = np.column_stack((X_cur[other_mask], Z_cur[other_mask]))
            balls_other.set_offsets(coords)

            return im, balls_other

        anim = animation.FuncAnimation(fig, update, frames=total_frames, interval=1000/fps, blit=False)
        writer = animation.FFMpegWriter(fps=fps, codec='libx264', extra_args=['-pix_fmt', 'yuv420p', '-crf', '18'])
        anim.save(output_mp4, writer=writer)
        plt.close(fig)
        print(f"[{idx}/5] Video success: {output_mp4}")

    print("\n All 5 shots are done!")

if __name__ == '__main__':
    main()
