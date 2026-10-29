#!/usr/bin/env python3
"""
Lagrangian sim (wave_lagrangian.mp4)
- only one particle (ball on the middle) is shown.
- Same grid: 216x216 px as the eulerian.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Parameters ---
fps = 60
duration = 3.5 # second
total_frames = int(fps * duration)

L = 12.0
x_tracked_eq = 6.0 # location of ball apparatus?

# wave parameters
c = 2.4
wavelength = 2.8
k = 2 * np.pi / wavelength
sigma = 1.25

# Wave or some shi bro just i grown tired of writing these comments on most basic code in earth like dont people understand what does line does? But Ferat you got to write more explicitly if you want to work with "these" people. I mean no i dont want to work with someone who cannot tell whether this code creates a black hole or just a basic calculation of wave placeement. Academia doesnt pay shi anyway. If i do that i do exactly as i wanted. for years, the same phrase has been echoing in my mind. Just let me handle this okay. I dont plan to stay for long anyway. let me do my job and leave
# So at t = 1.4 second wave will be on the ball
x_start = x_tracked_eq - c * 1.4 # = 6.0 - 3.36 = 2.64

# displacement amplitude (increased.)
A_disp = 0.55

# plot fig as always
fig = plt.figure(figsize=(15.0, 4.5), facecolor='white', dpi=120)
ax = fig.add_axes([0, 0, 1, 1], facecolor='white')

y_height = 1.2
ax.set_xlim(0, L)
ax.set_ylim(-y_height, y_height)
ax.axis('off')

# --- same grid (216x216 px, 2.5) ---
dy_grid = 0.96
dx_grid = 1.44

# 1. horizonal grid
y_lines = [-dy_grid, 0.0, dy_grid]
for y_val in y_lines:
    is_center = abs(y_val) < 1e-4
    lw = 1.5 if is_center else 0.9
    color = '#94A3B8' if is_center else '#CBD5E1'
    ax.axhline(y_val, color=color, lw=lw, zorder=1)

# 2. vertical
x_lines = [x_tracked_eq + k * dx_grid for k in range(-5, 6)]
for x_val in x_lines:
    if 0.0 <= x_val <= L:
        is_center = abs(x_val - x_tracked_eq) < 1e-4
        lw = 1.5 if is_center else 0.9
        color = '#94A3B8' if is_center else '#CBD5E1'
        ax.axvline(x_val, color=color, lw=lw, zorder=1)

# particle anchor
ax.axvline(x_tracked_eq, color='#CBD5E1', ls='--', lw=1.8, zorder=2)

# oscillation interval border (|---| if you not literate or some shi)
ax.plot(
    [x_tracked_eq - A_disp, x_tracked_eq + A_disp], 
    [-0.40, -0.40], 
    color='#F87171', lw=2.2, ls='-', zorder=3
)
ax.plot([x_tracked_eq - A_disp, x_tracked_eq - A_disp], [-0.52, -0.28], color='#F87171', lw=1.8, zorder=3)
ax.plot([x_tracked_eq + A_disp, x_tracked_eq + A_disp], [-0.52, -0.28], color='#F87171', lw=1.8, zorder=3)

# only one ball (Lagrangian yk)
ball_tracked = ax.scatter(
    [x_tracked_eq], 
    [0.0], 
    s=260, 
    c='#DC2626', 
    edgecolors='#991B1B', 
    linewidth=2.2, 
    zorder=6
)

# particle vel vector (arrow)
vel_arrow = ax.annotate(
    '', 
    xy=(x_tracked_eq, 0.40), 
    xytext=(x_tracked_eq, 0.40),
    arrowprops=dict(arrowstyle="->", color='#DC2626', lw=2.4, shrinkA=0, shrinkB=0),
    zorder=7
)


def update(frame):
    t = frame / fps
    x_center = x_start + c * t
    
    # particle location and displacement
    xi = x_tracked_eq - x_center
    env = np.exp(- (xi**2) / (2 * sigma**2))
    u_val = - A_disp * env * np.sin(k * xi)
    
    # vel (v = du/dt = -c * du/dx)
    du_dx = - A_disp * env * ( - (xi / sigma**2) * np.sin(k * xi) + k * np.cos(k * xi) )
    v_val = - c * du_dx
    
    # momentary location of ball apparatus? garfield? huh?
    x_current = x_tracked_eq + u_val
    ball_tracked.set_offsets([[x_current, 0.0]])
    
    # vel arrow
    arrow_scale = 0.35
    arrow_len = np.clip(v_val * arrow_scale, -0.75, 0.75)
    
    # vel small so no arrow.
    if abs(arrow_len) < 0.03:
        vel_arrow.xy = (x_current, 0.40)
        vel_arrow.set_position((x_current, 0.40))
    else:
        vel_arrow.xy = (x_current + arrow_len, 0.40)
        vel_arrow.set_position((x_current, 0.40))
    
    return ball_tracked, vel_arrow

anim = animation.FuncAnimation(fig, update, frames=total_frames, interval=1000/fps, blit=False)

output_mp4 = './wave_lagrangian.mp4'
writer = animation.FFMpegWriter(fps=fps, codec='libx264', extra_args=['-pix_fmt', 'yuv420p', '-crf', '18'])
anim.save(output_mp4, writer=writer)
plt.close()
print(f"Lagrangian done bro. You good as always: {output_mp4}")
