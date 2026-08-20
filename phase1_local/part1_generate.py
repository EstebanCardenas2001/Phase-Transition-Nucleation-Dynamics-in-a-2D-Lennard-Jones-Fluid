import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def init_system(N, L, T0):
    grid = int(np.ceil(np.sqrt(N)))
    spacing = L / grid
    x = np.linspace(spacing/2, L - spacing/2, grid)
    xv, yv = np.meshgrid(x, x)
    pos = np.column_stack((xv.ravel(), yv.ravel()))[:N]
    vel = np.random.normal(0, np.sqrt(T0), (N, 2))
    vel -= np.mean(vel, axis=0)
    return pos, vel

def compute_forces(pos, L):
    delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    delta -= L * np.round(delta / L)
    dist_sq = np.sum(delta**2, axis=-1)
    
    # Prevent Infinity * 0 errors
    np.fill_diagonal(dist_sq, 1.0)
    
    inv_r2 = 1.0 / dist_sq
    inv_r6 = inv_r2**3
    inv_r12 = inv_r6**2
    
    force_mag = 24.0 * (2.0 * inv_r12 - inv_r6) * inv_r2
    forces = np.sum(force_mag[:, :, np.newaxis] * delta, axis=1)
    pe = np.sum(4.0 * (inv_r12 - inv_r6)) / 2.0
    
    W_ij = 24.0 * (2.0 * inv_r12 - inv_r6)
    np.fill_diagonal(W_ij, 0.0)
    virial_sum = np.sum(W_ij) / 2.0
    
    return forces, pe, virial_sum

def velocity_verlet_step(pos, vel, forces, L, dt):
    vel += 0.5 * forces * dt
    pos = (pos + vel * dt) % L
    new_forces, pe, virial_sum = compute_forces(pos, L)
    vel += 0.5 * new_forces * dt
    return pos, vel, new_forces, pe, virial_sum

def apply_andersen_thermostat(vel, T_target, collision_freq, dt):
    N = vel.shape[0]
    mask = np.random.rand(N) < (collision_freq * dt)
    num_collided = np.sum(mask)
    if num_collided > 0:
        vel[mask] = np.random.normal(0, np.sqrt(T_target), (num_collided, 2))
    return vel

def run_part1_massive():
    N = 4000  # Massive particle count
    rho = 0.3
    L = np.sqrt(N / rho)
    dt = 0.005
    T_ladder = np.linspace(0.60, 0.35, 12)
    
    trajectory, actual_T, target_T, energies = [], [], [], []
    pressures, order_params, coordinations = [], [], []
    r_c = 1.5
    
    pos, vel = init_system(N, L, T_ladder[0])
    forces, _, _ = compute_forces(pos, L)
    
    start_time = time.time()
    
    burn_in_steps = 6000
    record_interval = 40
    burn_in_frames = 0
    
    print(f"Phase 1/2: Burning in massive N={N} system using NumPy vectorization...")
    for step in range(burn_in_steps):
        pos, vel, forces, pe, virial = velocity_verlet_step(pos, vel, forces, L, dt)
        vel = apply_andersen_thermostat(vel, T_ladder[0], 0.5, dt)
        
        if step % record_interval == 0:
            ke = 0.5 * np.sum(vel**2)
            inst_T = ke / N
            inst_P = (N * inst_T / (L**2)) + (virial / (2.0 * L**2))
            
            delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            delta -= L * np.round(delta / L)
            dist_sq = np.sum(delta**2, axis=-1)
            np.fill_diagonal(dist_sq, np.inf)
            adjacency = (dist_sq < r_c**2).astype(int)
            
            graph = csr_matrix(adjacency)
            _, labels = connected_components(csgraph=graph, directed=False)
            _, counts = np.unique(labels, return_counts=True)
            coord_num = np.sum(adjacency, axis=1)
            
            trajectory.append(np.float16(pos))
            actual_T.append(inst_T)
            target_T.append(T_ladder[0])
            energies.append(pe / N)
            pressures.append(inst_P)
            order_params.append(np.max(counts) / N)
            coordinations.append(np.int8(coord_num))
            burn_in_frames += 1

    prod_steps_per_rung = 4000 
    print(f"Phase 2/2: Starting production cooling ladder...")
    for T in T_ladder:
        print(f"  Targeting T = {T:.3f}...")
        for step in range(prod_steps_per_rung):
            pos, vel, forces, pe, virial = velocity_verlet_step(pos, vel, forces, L, dt)
            vel = apply_andersen_thermostat(vel, T, 0.5, dt)
            
            if step % record_interval == 0:
                ke = 0.5 * np.sum(vel**2)
                inst_T = ke / N
                inst_P = (N * inst_T / (L**2)) + (virial / (2.0 * L**2))
                
                delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
                delta -= L * np.round(delta / L)
                dist_sq = np.sum(delta**2, axis=-1)
                np.fill_diagonal(dist_sq, np.inf)
                adjacency = (dist_sq < r_c**2).astype(int)
                
                graph = csr_matrix(adjacency)
                _, labels = connected_components(csgraph=graph, directed=False)
                _, counts = np.unique(labels, return_counts=True)
                coord_num = np.sum(adjacency, axis=1)
                
                trajectory.append(np.float16(pos))
                actual_T.append(inst_T)
                target_T.append(T)
                energies.append(pe / N)
                pressures.append(inst_P)
                order_params.append(np.max(counts) / N)
                coordinations.append(np.int8(coord_num))

    output_file = 'part1_ladder_massive.npz'
    np.savez_compressed(output_file, 
                        traj=np.array(trajectory), 
                        T_act=np.array(actual_T), 
                        T_targ=np.array(target_T), 
                        E=np.array(energies),
                        P=np.array(pressures),
                        m=np.array(order_params), 
                        coord=np.array(coordinations),
                        burn_in_frames=burn_in_frames,
                        L=L, N=N)
    
    elapsed = (time.time() - start_time) / 60
    print(f"Massive Simulation complete in {elapsed:.1f} minutes. Saved to {output_file}")

if __name__ == "__main__":
    run_part1_massive()