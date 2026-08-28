import torch
import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# --- 1. GPU Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Executing on: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# --- 2. Thermodynamic Parameters (Optimized for 24h runtime) ---
N = 32768                # 181x181 lattice (Double the original 16k)
rho = 0.3                # Number density
L = float(np.sqrt(N / rho)) # Box length
dt = 0.005               # Time step
save_freq = 60           # Save frequency (Yields exactly 1,000 frames)

# Cooling ladder schedule (60,000 total steps)
burn_in_steps = 10000
prod_steps = 50000
total_steps = burn_in_steps + prod_steps

T_init = 2.0             # Hot gas phase
T_final = 0.0            # Deep freeze/crystallization phase
nu = 1.0                 # High collision frequency to extract latent heat

print(f"System: N={N}, L={L:.2f}, Total Steps={total_steps}")

# --- 3. Initialize Tensors on GPU ---
grid_pts = int(np.ceil(np.sqrt(N)))
spacing = L / grid_pts
x = torch.linspace(spacing/2, L - spacing/2, grid_pts, device=device)
y = torch.linspace(spacing/2, L - spacing/2, grid_pts, device=device)
grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
pos = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)[:N]

vel = torch.randn((N, 2), device=device) * torch.sqrt(torch.tensor(T_init, device=device))
vel -= torch.mean(vel, dim=0)

# --- 4. Chunked GPU Force Engine (OOM Prevention) ---
def compute_forces(p, box_length, return_graph=False):
    forces = torch.zeros_like(p)
    total_pe = torch.tensor(0.0, device=device)
    
    # 1024 chunk size keeps VRAM safely around ~3GB
    chunk_size = 1024  
    
    if return_graph:
        edges_i = []
        edges_j = []
        coordination = torch.zeros(N, dtype=torch.int32, device=device)
        
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        p_chunk = p[i:end]
        
        # CORRECTED: p_chunk (i) - p (j) restores the correct repulsive physics
        dx = p_chunk.unsqueeze(1) - p.unsqueeze(0)
        
        dx.sub_(box_length * torch.round(dx / box_length))
        r2 = torch.sum(dx**2, dim=-1)
        
        # Mask self-interactions
        eye_mask = torch.zeros_like(r2, dtype=torch.bool)
        eye_mask[:, i:end] = torch.eye(end - i, device=device, dtype=torch.bool)
        r2.masked_fill_(eye_mask, float('inf'))
        
        # SAFETY CLAMP: Prevents float32 division by zero
        r2 = torch.clamp(r2, min=0.01)
        
        mask = r2 < 6.25 
        
        r2_inv = torch.zeros_like(r2)
        r2_inv[mask] = 1.0 / r2[mask]
        
        r6_inv = r2_inv ** 3
        r12_inv = r6_inv ** 2
        
        f_mag = 48.0 * (r12_inv * r2_inv - 0.5 * r6_inv * r2_inv)
        f_mag[~mask] = 0.0
        
        forces[i:end] = torch.sum(f_mag.unsqueeze(-1) * dx, dim=1)
        
        pe = 4.0 * (r12_inv - r6_inv)
        pe[~mask] = 0.0
        total_pe += 0.5 * torch.sum(pe)
        
        if return_graph:
            bond_mask = r2 < 2.25
            coordination[i:end] = torch.sum(bond_mask, dim=1).to(torch.int32)
            
            idx_chunk, idx_n = torch.where(bond_mask)
            edges_i.append(idx_chunk + i)
            edges_j.append(idx_n)
            
    if return_graph:
        return forces, total_pe, torch.cat(edges_i), torch.cat(edges_j), coordination
    
    return forces, total_pe

# --- 5. Data Tracking Arrays ---
traj_out = []
T_act_out = []
T_targ_out = []
E_out = []
m_out = []
coord_out = []

forces, pot_E = compute_forces(pos, L, return_graph=False)

# --- 6. Integration Loop ---
start_time = time.time()

for step in range(total_steps):
    if step < burn_in_steps:
        T_target = T_init
    else:
        progress = (step - burn_in_steps) / prod_steps
        T_target = T_init - progress * (T_init - T_final)
        T_target = max(T_target, 0.0)
        
    pos = pos + vel * dt + 0.5 * forces * dt**2
    pos = torch.remainder(pos, L)
    vel_half = vel + 0.5 * forces * dt
    
    check_graph = (step % save_freq == 0)
    
    if check_graph:
        forces, pot_E, e_i, e_j, coord = compute_forces(pos, L, return_graph=True)
    else:
        forces, pot_E = compute_forces(pos, L, return_graph=False)
        
    vel = vel_half + 0.5 * forces * dt
    
    collision_mask = torch.rand(N, device=device) < (nu * dt)
    num_collisions = collision_mask.sum().item()
    if num_collisions > 0:
        if T_target > 0.0:
            thermal_vel = torch.randn((num_collisions, 2), device=device) * np.sqrt(T_target)
        else:
            thermal_vel = torch.zeros((num_collisions, 2), device=device)
        vel[collision_mask] = thermal_vel
    
    if check_graph:
        kin_E = 0.5 * torch.sum(vel**2)
        T_actual = (kin_E / N).item()
        E_actual = (pot_E / N).item()
        
        edges_i_cpu = e_i.cpu().numpy()
        edges_j_cpu = e_j.cpu().numpy()
        vals = np.ones_like(edges_i_cpu)
        graph = csr_matrix((vals, (edges_i_cpu, edges_j_cpu)), shape=(N, N))
        
        n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        _, counts = np.unique(labels, return_counts=True)
        largest_cluster = np.max(counts) if len(counts) > 0 else 1
        m_fraction = largest_cluster / N
        
        traj_out.append(pos.cpu().numpy().astype(np.float16))
        T_act_out.append(T_actual)
        T_targ_out.append(T_target)
        E_out.append(E_actual)
        m_out.append(m_fraction)
        coord_out.append(coord.cpu().numpy().astype(np.int8))
        
        elapsed = time.time() - start_time
        
        if step == 0:
            fps = 0.0
            eta_hours = 0.0
            eta_mins = 0.0
        else:
            fps = step / elapsed
            steps_left = total_steps - step
            eta_seconds = steps_left / fps
            eta_hours = eta_seconds / 3600
            eta_mins = (eta_seconds % 3600) / 60
        
        phase = "BURN-IN" if step < burn_in_steps else "PROD"
        print(f"[{phase}] Step {step:06d}/{total_steps} | T: {T_actual:.2f} | m: {m_fraction:.3f} | FPS: {fps:.1f} | ETA: {eta_hours:.1f}h {eta_mins:.0f}m")

# --- 7. Save High-Performance Output ---
print("\nSaving massive trajectory data...")
np.savez_compressed(
    'part2_ladder_massive.npz',
    traj=np.array(traj_out),
    T_act=np.array(T_act_out),
    T_targ=np.array(T_targ_out),
    E=np.array(E_out),
    m=np.array(m_out),
    coord=np.array(coord_out),
    L=L,
    N=N,
    burn_in_frames=burn_in_steps // save_freq
)
print(f"Simulation complete. File saved as 'part2_ladder_massive.npz'. Total Time: {(time.time()-start_time)/3600:.2f} hrs.")