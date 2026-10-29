#!/usr/bin/env python3
"""
just wave sim and few arrows
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

fps = 60
duration = 6.0 # sec
total_frames = int(fps * duration)

L = 12.0             # length of the line
N_balls = 75         # Number of Atom (ball)
x_eq = np.linspace(0.4, L - 0.4, N_balls)
dx_eq = x_eq[1] - x_eq[0]

# wave parameters
c = 2.4              # Wave propagating velocity
wavelength = 2.2     # Wave lambda
k = 2 * np.pi / wavelength
omega = k * c

# Wave soliton or envelope ion now i mean no soliton math which is unprecedentedly cool right?
sigma = 1.35
x_start = -2.8       # wave start with left at t=0 (all stationary/white)

# disp amp
A_disp = dx_eq * 0.42

# background imshow x resolution
nx_bg = 600
x_bg = np.linspace(0, L, nx_bg)

# Blue-White-Red color map
# 0 is pure white (#FFFFFF)
cmap_wave = LinearSegmentedColormap.from_list(
    'clean_bwr', 
    [
        (0.00, '#2563EB'), # trough (-1, blue)
        (0.25, '#93C5FD'), # blueish
        (0.50, '#FFFFFF'), # zero (0, white)
        (0.75, '#FCA5A5'), # redish
        (1.00, '#DC2626')  # crest (+1, red)
    ]
)

# Matplotlib fig plot
fig, ax = plt.subplots(figsize=(15, 4.2), facecolor='white', dpi=120)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# axis
y_height = 1.1
ax.set_xlim(0, L)
ax.set_ylim(-y_height, y_height)
ax.axis('off')

# stat axis (minimal)
ax.axhline(0, color='#E2E8F0', lw=1.5, zorder=2)

# background wave (imshow)
init_density = np.zeros((1, nx_bg))
im = ax.imshow(
    init_density, 
    extent=[0, L, -y_height, y_height], 
    aspect='auto', 
    cmap=cmap_wave, 
    vmin=-1.0, 
    vmax=1.0, 
    alpha=0.65, 
    zorder=1
)

# balls (born-oppenheimer)
balls = ax.scatter(
    x_eq, 
    np.zeros_like(x_eq), 
    s=65, 
    c='#1E293B', 
    edgecolors='#0F172A', 
    linewidth=1.2, 
    zorder=5
)

# anchor
mid_idx = N_balls // 2
ax.plot(
    x_eq[mid_idx], 0, 
    marker='o', markersize=9, 
    fillstyle='none', markeredgecolor='#94A3B8', markeredgewidth=1.2, 
    ls='none', zorder=3
)

plt.tight_layout(pad=0.1)

def calc_fields(t):
    """
    Belirli bir t anı için arka plan dalga genliğini ve parçacık deplasmanlarını hesaplar.
    """
    x_center = x_start + c * t
    
    # density
    xi_bg = x_bg - x_center
    env_bg = np.exp(- (xi_bg**2) / (2 * sigma**2))
    density = env_bg * np.cos(k * xi_bg) # [-1, 1] arası
    
    # Particle Displacemnent u(x, t)
    # crest (cos > 0) compression (closer balls in winter: du/dx < 0)
    xi_p = x_eq - x_center
    env_p = np.exp(- (xi_p**2) / (2 * sigma**2))
    u_vals = - A_disp * env_p * np.sin(k * xi_p)
    
    return density, u_vals

def update(frame):
    t = frame / fps
    density, u_vals = calc_fields(t)
    
    # update background
    im.set_data(density[np.newaxis, :])
    
    # update balls (only x axis)
    x_current = x_eq + u_vals
    balls.set_offsets(np.column_stack((x_current, np.zeros_like(x_current))))
    
    return im, balls

anim = animation.FuncAnimation(fig, update, frames=total_frames, interval=1000/fps, blit=True)

output_mp4 = './wave_1d_strips.mp4'
writer = animation.FFMpegWriter(fps=fps, codec='libx264', extra_args=['-pix_fmt', 'yuv420p', '-crf', '18'])
anim.save(output_mp4, writer=writer)
plt.close()
print(f"Video saved: {output_mp4}")
