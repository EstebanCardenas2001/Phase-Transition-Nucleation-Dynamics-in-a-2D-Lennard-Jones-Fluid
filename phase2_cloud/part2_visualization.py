import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch

mpl.rcParams['animation.embed_limit'] = 2000.0

# --- 1. Load Massive Dataset ---
data = np.load('part2_ladder_massive.npz')

# Cast float16 back to float32 for Matplotlib compatibility
trajectory = data['traj'].astype(np.float32)
T_act = data['T_act']
T_targ = data['T_targ']
energies = data['E']
order_params = data['m']
coordinations = data['coord']
L = float(data['L'])
N = int(data['N'])
burn_in_frames = int(data['burn_in_frames'])

# Utilize all frames for a smooth video
num_frames = trajectory.shape[0]
num_prod_frames = num_frames - burn_in_frames
time_axis = np.arange(num_prod_frames)

print(f"Rendering {num_frames} total frames ({burn_in_frames} burn-in, {num_prod_frames} production) for N={N}...")

# --- 2. Calculate Plot Limits (Production Only) ---
prod_T = T_act[burn_in_frames:]
prod_E = energies[burn_in_frames:]
prod_m = order_params[burn_in_frames:]

t_min, t_max = np.min(prod_T), np.max(prod_T)
e_min, e_max = np.min(prod_E), np.max(prod_E)
e_pad = (e_max - e_min) * 0.1

# Dynamic scaling for the Order Parameter axis
m_max = np.max(prod_m) * 1.2 if np.max(prod_m) > 0 else 0.1

# --- 3. Dashboard Layout ---
plt.ioff()
fig = plt.figure(figsize=(18, 12), dpi=200)

gs = GridSpec(2, 6, height_ratios=[1.5, 1], wspace=0.8, hspace=0.3)
fig.subplots_adjust(right=0.90, top=0.90, bottom=0.08, left=0.08)

# Top Left: Full Physical MD View
ax_sim = fig.add_subplot(gs[0, 0:3])
ax_sim.set_xlim(0, L)
ax_sim.set_ylim(0, L)
ax_sim.set_aspect('equal')
ax_sim.set_xticks([])
ax_sim.set_yticks([])
ax_sim.set_facecolor('black')
ax_sim.set_title(f'Macroscopic Global Domain (N={N})', fontsize=14, pad=15)

zoom_w = L / 5 
zoom_min = (L / 2) - (zoom_w / 2)
zoom_max = (L / 2) + (zoom_w / 2)

zoom_rect = Rectangle((zoom_min, zoom_min), zoom_w, zoom_w, 
                      linewidth=1.5, edgecolor='white', facecolor='none', linestyle='--', alpha=0.7)
ax_sim.add_patch(zoom_rect)

# Top Right: Zoomed Physical View
ax_zoom = fig.add_subplot(gs[0, 3:6])
ax_zoom.set_xlim(zoom_min, zoom_max)
ax_zoom.set_ylim(zoom_min, zoom_max)
ax_zoom.set_aspect('equal')
ax_zoom.set_xticks([])
ax_zoom.set_yticks([])
ax_zoom.set_facecolor('black')
ax_zoom.set_title('Magnified Core View', fontsize=14, pad=15)

for (x_sim, y_sim) in [(zoom_min, zoom_min), (zoom_max, zoom_min), (zoom_min, zoom_max), (zoom_max, zoom_max)]:
    con = ConnectionPatch(xyA=(x_sim, y_sim), coordsA=ax_sim.transData,
                          xyB=(x_sim, y_sim), coordsB=ax_zoom.transData,
                          color='orange', linestyle='--', linewidth=1.0, alpha=1)
    fig.add_artist(con)

status_text = fig.text(0.5, 0.94, '', ha='center', va='center', fontsize=16, fontweight='bold', 
                       color='red', bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', pad=8))

# Bottom Left: Bimodal Density Histogram
ax_hist = fig.add_subplot(gs[1, 0:2])
ax_hist.set_xlim(-0.5, 8.5)
ax_hist.set_ylim(0, N * 0.7)
ax_hist.set_xlabel('Local Coordination Number (Density)')
ax_hist.set_ylabel('Particle Count')
ax_hist.set_title('Phase Coexistence', fontsize=12)
ax_hist.grid(True, axis='y', linestyle='--', alpha=0.6)

cmap = cm.get_cmap('coolwarm')
bar_colors = [cmap(min(i / 6.0, 1.0)) for i in range(10)]
hist_bins = np.arange(-0.5, 10.5, 1.0)
hist_bars = ax_hist.bar(np.arange(10), np.zeros(10), width=0.8, color=bar_colors, edgecolor='black', alpha=0.9)

# Bottom Center: Energy Trace
ax_ener = fig.add_subplot(gs[1, 2:4])
ax_ener.set_xlim(0, num_prod_frames)
ax_ener.set_ylim(e_min - e_pad, e_max + e_pad)
ax_ener.set_ylabel('Potential Energy / $N$', color='blue')
ax_ener.set_xlabel('Production Frame Index')
ax_ener.grid(True, linestyle='--', alpha=0.6)

ax_temp = ax_ener.twinx()
ax_temp.set_ylim(-0.1, 2.2) 
ax_temp.set_ylabel('Temperature', color='red', labelpad=12)

# Bottom Right: Order Parameter
ax_order = fig.add_subplot(gs[1, 4:6])
ax_order.set_xlim(0, num_prod_frames)
ax_order.set_ylim(0, m_max) # Dynamically scaled axis
ax_order.set_xlabel('Production Frame Index')
ax_order.set_title('Structural Order Parameter', fontsize=12)
ax_order.grid(True, linestyle='--', alpha=0.6)

ax_order_right = ax_order.twinx()
ax_order_right.set_ylim(0, m_max) # Dynamically scaled axis
ax_order_right.set_ylabel('Largest Droplet Fraction ($m$)', color='purple', labelpad=12)
ax_order_right.set_yticks([])

# --- 4. Dynamic Artists ---
# Increased marker size (s=1.5) and solid opacity (alpha=1.0) for visibility
scatter = ax_sim.scatter(trajectory[0, :, 0], trajectory[0, :, 1], 
                         s=1.5, c=coordinations[0], cmap='coolwarm', 
                         vmin=0, vmax=6, alpha=1.0, edgecolors='none')

scatter_zoom = ax_zoom.scatter(trajectory[0, :, 0], trajectory[0, :, 1], 
                               s=8, c=coordinations[0], cmap='coolwarm', 
                               vmin=0, vmax=6, alpha=0.9, edgecolors='none')

line_ener, = ax_ener.plot([], [], color='blue', linewidth=1.5, label='Potential E')
line_targ, = ax_temp.plot([], [], color='black', linestyle='--', label='Target T')
line_act, = ax_temp.plot([], [], color='red', alpha=0.7, label='Actual T')
line_ord, = ax_order.plot([], [], color='purple', linewidth=1.5)

vlines = [ax_ener.axvline(0, color='gray', linestyle=':', alpha=0.8),
          ax_order.axvline(0, color='gray', linestyle=':', alpha=0.8)]

# --- 5. Phase-Aware Update Protocol ---
def update(frame):
    scatter.set_offsets(trajectory[frame])
    scatter.set_array(coordinations[frame])
    
    scatter_zoom.set_offsets(trajectory[frame])
    scatter_zoom.set_array(coordinations[frame])
    
    if frame < burn_in_frames:
        status_text.set_text("PHASE: BURN-IN (Thermalizing)")
        status_text.set_color('tomato')
        
        for bar in hist_bars:
            bar.set_height(0)
            
        line_ener.set_data([], [])
        line_targ.set_data([], [])
        line_act.set_data([], [])
        line_ord.set_data([], [])
        for vl in vlines:
            vl.set_visible(False)
    else:
        status_text.set_text("PHASE: PRODUCTION & COOLING")
        status_text.set_color('green')
        
        counts, _ = np.histogram(coordinations[frame], bins=hist_bins)
        for bar, count in zip(hist_bars, counts):
            bar.set_height(count)
            
        p_frame = frame - burn_in_frames
        curr_t_idx = time_axis[:p_frame+1]
        
        line_ener.set_data(curr_t_idx, energies[burn_in_frames : frame+1])
        line_targ.set_data(curr_t_idx, T_targ[burn_in_frames : frame+1])
        line_act.set_data(curr_t_idx, T_act[burn_in_frames : frame+1])
        line_ord.set_data(curr_t_idx, order_params[burn_in_frames : frame+1])
        
        for vl in vlines:
            vl.set_visible(True)
            vl.set_xdata([p_frame, p_frame])
        
    return scatter, scatter_zoom, line_ener, line_targ, line_act, line_ord, *hist_bars, *vlines

ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=30, blit=True)

print("Rendering hardware-accelerated 4K MP4 (Expected duration: ~33s)...")
try:
    ani.save(
        'massive_32k_dashboard.mp4', 
        writer='ffmpeg', 
        fps=30, 
        dpi=200,  # 18x12 @ 200 DPI = 3600x2400 (Clean 4K within hardware limits)
        extra_args=[
            '-vcodec', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '17',            # Visually lossless
            '-preset', 'fast',       # Prevents playback decoder lag
            '-profile:v', 'high',
            '-level', '5.1',
            '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2'  # Ensures even pixel dimensions
        ]
    )
    print("Successfully saved 'massive_32k_dashboard.mp4'!")
except Exception as e:
    print(f"FFmpeg render failed: {e}")
    ani.save('massive_32k_dashboard.gif', writer='pillow', fps=30, dpi=120)
    print("Saved fallback GIF.")