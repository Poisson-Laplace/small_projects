#!/usr/bin/env python3
"""
messed up figure
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fps = 60
duration = 6.0 # saniye
total_frames = int(fps * duration)
L = 12.0
N_balls = 75
x_eq = np.linspace(0.4, L - 0.4, N_balls)
dx_eq = x_eq[1] - x_eq[0]
c = 2.4
wavelength = 2.0
k = 2 * np.pi / wavelength
omega = k * c

# Gaussian envelope
sigma = 1.3
x_start = -2.8
A_wave = 0.85 
A_disp = dx_eq * 0.42
nx_dense = 900
x_dense = np.linspace(0, L, nx_dense)

# Matplotlib
fig, ax = plt.subplots(figsize=(15, 4.5), facecolor='white', dpi=120)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.set_xlim(0, L)
ax.set_ylim(-1.35, 1.35)
ax.axis('off')
ax.axhline(0, color='#E2E8F0', lw=1.5, zorder=1)
fill_red = ax.fill_between(x_dense, 0, 0, color='#EF4444', alpha=0.35, zorder=2)
fill_blue = ax.fill_between(x_dense, 0, 0, color='#3B82F6', alpha=0.35, zorder=2)
line_wave, = ax.plot(x_dense, np.zeros_like(x_dense), color='#475569', lw=1.8, zorder=3)

# Born-Oppenheimer
balls = ax.scatter(x_eq, np.zeros_like(x_eq), s=65, c='#1E293B', edgecolors='#0F172A', linewidth=1.2, zorder=5)
mid_idx = N_balls // 2
x_mid_eq = x_eq[mid_idx]
ax.plot(x_mid_eq, 0, marker='o', markersize=9, fillstyle='none', markeredgecolor='#94A3B8', markeredgewidth=1.2, ls='none', zorder=4)

plt.tight_layout(pad=0.2)

def wave_profile(x_arr, t):
    x_center = x_start + c * t
    xi = x_arr - x_center
    envelope = np.exp(- (xi**2) / (2 * sigma**2))
    return A_wave * envelope * np.cos(k * xi)

def particle_displacement(x_arr, t):
    x_center = x_start + c * t
    xi = x_arr - x_center
    envelope = np.exp(- (xi**2) / (2 * sigma**2))
    return - A_disp * envelope * np.sin(k * xi)

def update(frame):
    global fill_red, fill_blue
    t = frame / fps
    W = wave_profile(x_dense, t)
    line_wave.set_ydata(W)
    fill_red.remove()
    fill_blue.remove()
    
    W_pos = np.where(W > 0, W, 0)
    W_neg = np.where(W < 0, W, 0)
    
    fill_red = ax.fill_between(x_dense, 0, W_pos, color='#EF4444', alpha=0.32, zorder=2)
    fill_blue = ax.fill_between(x_dense, 0, W_neg, color='#3B82F6', alpha=0.32, zorder=2)
    u_vals = particle_displacement(x_eq, t)
    x_current = x_eq + u_vals
    balls.set_offsets(np.column_stack((x_current, np.zeros_like(x_current))))
    
    return line_wave, balls

anim = animation.FuncAnimation(fig, update, frames=total_frames, interval=1000/fps, blit=False)

output_mp4 = './wave_1d_ballz.mp4'
writer = animation.FFMpegWriter(fps=fps, codec='libx264', extra_args=['-pix_fmt', 'yuv420p', '-crf', '18'])
anim.save(output_mp4, writer=writer)
plt.close()
print(f"Video saved: {output_mp4}")
