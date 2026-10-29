#!/usr/bin/env python3
"""
shows what is eularian perspective in continuum mechanics (wave_eulerian.mp4). May be used in both fluid mechanics or wave physics or i actually dont know like i just dont feel good at this point. Let me take a break. I have to go to thte gym or walk a little 
- compatible with lagrangian one. twin animasyon.
- Resolution: 1800x540 piksel.
- x axis: [0, 12.0], z axis (y): [-1.2, 1.2]
- Background grid: spatially stationary (216x216 px square, 2.5 times vertically).
- Ball: Moves with wave (x_cur = x_tracked_eq + u(t)).
- Velocity Vector: On top of the particle (y = 0.40), also moves with particle and shows vel.
- Same parameters: A_disp = 0.65, t_peak = 1.4, c = 2.4, duration = 3.5 s.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Parameters (same with lagrangian one) ---
fps = 60
duration = 3.5 # second
total_frames = int(fps * duration)

# Res and axis (1800x540 px)
fig_w, fig_h = 15.0, 4.5
dpi = 120
L = 12.0
x_tracked_eq = 6.0
y_height = 1.2

# 216x216 pix sq grid spacing
dy_grid = 0.96 # vertically 2.5 (540 / 2.5 = 216 px)
dx_grid = 1.44 # horizontally 216 px

# wave parameters
c = 2.4
wavelength = 2.8
k = 2 * np.pi / wavelength
sigma = 1.25
t_peak = 1.4

# displacement amplitude
A_disp = 0.65

# plot fig - no space on sides
fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white', dpi=dpi)
ax = fig.add_axes([0, 0, 1, 1], facecolor='white')

ax.set_xlim(0, L)
ax.set_ylim(-y_height, y_height)
ax.axis('off')

# 1. horizontal grid lines (z axis 2.5 square)
y_lines = [-dy_grid, 0.0, dy_grid]
for y_val in y_lines:
    is_center = abs(y_val) < 1e-4
    lw = 1.5 if is_center else 0.9
    color = '#94A3B8' if is_center else '#CBD5E1'
    ax.axhline(y_val, color=color, lw=lw, zorder=1)

# 2. Vertical grid lines (stationary)
for m in range(-5, 6):
    x_val = x_tracked_eq + m * dx_grid
    if 0.0 <= x_val <= L:
        is_center = (m == 0)
        lw = 1.5 if is_center else 0.9
        color = '#94A3B8' if is_center else '#CBD5E1'
        ls = '--' if is_center else '-'
        ax.plot([x_val, x_val], [-y_height, y_height], color=color, lw=lw, ls=ls, zorder=1)

# 3. Ball (initially x = 6.0, after the wave it oscillates, s = 260)
ball = ax.scatter(
    [x_tracked_eq], 
    [0.0], 
    s=260, 
    c='#DC2626', 
    edgecolors='#991B1B', 
    linewidth=2.2, 
    zorder=6
)

# 4. Vel vectory arrow (y = 0.40, moves with particle also)
vel_arrow = ax.annotate(
    '', 
    xy=(x_tracked_eq, 0.40), 
    xytext=(x_tracked_eq, 0.40),
    arrowprops=dict(arrowstyle="->", color='#DC2626', lw=2.4, shrinkA=0, shrinkB=0),
    zorder=7
)

def calc_motion(t):
    dt_eff = t - t_peak
    tau = - c * dt_eff
    env = np.exp(- (tau**2) / (2 * sigma**2))
    
    # displacement u(t)
    u = - A_disp * env * np.sin(k * tau)
    
    # velocity v = du/dt = -c * du/dx
    du_dx = - A_disp * env * ( - (tau / sigma**2) * np.sin(k * tau) + k * np.cos(k * tau) )
    v = - c * du_dx
    
    return u, v

def update(frame):
    t = frame / fps
    u, v = calc_motion(t)
    
    # 1. location of the ball
    x_current = x_tracked_eq + u
    ball.set_offsets([[x_current, 0.0]])
    
    # 2. update the velocity vector (on top of the ball | with both direction and magnitude oh yeahhh!!)
    arrow_scale = 0.35
    arrow_len = np.clip(v * arrow_scale, -0.85, 0.85)
    
    if abs(arrow_len) < 0.03:
        vel_arrow.xy = (x_current, 0.40)
        vel_arrow.set_position((x_current, 0.40))
    else:
        vel_arrow.xy = (x_current + arrow_len, 0.40)
        vel_arrow.set_position((x_current, 0.40))
        
    return ball, vel_arrow

anim = animation.FuncAnimation(fig, update, frames=total_frames, interval=1000/fps, blit=False)

output_mp4 = './wave_eulerian.mp4'
writer = animation.FFMpegWriter(fps=fps, codec='libx264', extra_args=['-pix_fmt', 'yuv420p', '-crf', '18'])
anim.save(output_mp4, writer=writer)
plt.close()
print(f"Eulerian done bro. you good: {output_mp4}")
