# Phase Transition & Nucleation Dynamics in a 2D Lennard-Jones Fluid


<div align="center">
  <img src="outputs/dashboard_preview.gif" alt="MD Phase Transition Simulation" width="100%">
</div>
[![Macroscopic 32k Simulation Phase Transition](<img width="1360" height="944" alt="Screenshot" src="https://github.com/user-attachments/assets/66cd0fa3-4698-46c1-9743-0584b61f54bc" />
)](https://youtu.be/kXBRlVEVq5Y
)
---

## Project Overview

This project explores the non-equilibrium thermodynamics of a 2D Lennard-Jones fluid undergoing a phase transition. By simulating a continuous cooling ladder, the system spontaneously breaks symmetry, demonstrating droplet nucleation and gas-liquid phase coexistence. 

The project is structured into two distinct engineering phases: building a robust, fully vectorized local physics engine, and subsequently scaling the architecture to cloud GPUs to approach the macroscopic thermodynamic limit.

## Phase 1: Local Prototyping & Physics Engine

The first phase focuses on algorithm design and building a ground-up Molecular Dynamics (MD) engine without relying on external simulation packages (like LAMMPS). The engine was designed to maximize single-node CPU performance using matrix vectorization.

*   **Interaction Potential:** Standard Lennard-Jones 12-6 potential $V(r) = 4\epsilon [(\frac{\sigma}{r})^{12} - (\frac{\sigma}{r})^6]$ with periodic boundary conditions (PBC).
*   **Vectorized Force Matrix:** The $O(N^2)$ pairwise distance calculations are strictly vectorized through NumPy to leverage Apple Silicon's Accelerate framework, calculating tens of millions of pairwise interactions per time step.
*   **Thermodynamics:** Uses a Velocity Verlet integrator coupled with an Andersen thermostat. The system undergoes a strict burn-in phase to thermalize the initial lattice, followed by a stepwise production cooling ladder to induce nucleation.
*   **Order Parameter Tracking:** Calculates the local coordination number via a sparse adjacency matrix to identify connected components, allowing real-time tracking of the largest droplet fraction ($m$).

## The Visualization Dashboard

To extract physical meaning from the raw trajectory data, the project includes a custom 5-panel Matplotlib dashboard that renders the dynamics alongside real-time thermodynamic observables:

| Panel | Insight |
| :--- | :--- |
| **Global Domain** | Visualizes the entire simulation box subject to PBC. Particles are color-coded by their local coordination number. |
| **Magnified Core** | Uses a dynamic tracking algorithm to anchor the camera to the densest cluster, providing a stable, zoomed-in view of droplet condensation. |
| **Bimodal Density** | A real-time histogram proving phase coexistence. As the system cools, the single peak splits into a low-density gas phase and a high-density liquid phase. |
| **Energy & Temp** | Tracks the target vs. actual temperature and the corresponding drops in potential energy as bonds form. |
| **Order Parameter** | Plots the fraction of particles belonging to the largest single cluster, capturing the exact moment of symmetry breaking. |

# Phase 2: Macroscopic Phase Transition & Nucleation Dynamics

This phase scales the 2D Lennard-Jones fluid simulation to macroscopic dimensions ($N = 32,768$ particles) to observe true first-order phase transitions, latent heat extraction, and the nucleation of a polycrystalline solid. 

By drastically increasing the system size, we transition from observing isolated microscopic droplet formations to simulating a massive, interconnected thermodynamic ensemble.

##  Computational Physics: The GPU Force Engine

Scaling the system to 32,768 particles introduces a severe $O(N^2)$ computational bottleneck. Calculating the pairwise distance matrix for periodic boundaries instantaneously requires over 14 GB of VRAM, which exceeds standard hardware limits and causes memory fragmentation.

To bypass this and achieve maximum hardware saturation on an NVIDIA Tesla T4, the force engine in `part2_generate_gpu.py` was completely rewritten using a **Chunked VRAM algorithm**:
* **Memory Optimization:** Slices the particle tensor into chunks of 1024, keeping the peak VRAM footprint safely below 3 GB.
* **In-Place Operations:** Utilizes `.sub_()` to execute periodic boundary vector subtractions directly in memory, preventing temporary tensor bloat.
* **100% Compute Saturation:** Feeds the CUDA cores exactly as much data as they can mathematically process per cycle, maintaining a constant 100% GPU utilization without memory overhead.

## Thermodynamics & Latent Heat

The simulation employs a cooling ladder over 60,000 integration steps ($10,000$ burn-in, $50,000$ production), chilling the system from a hot gas phase ($T=2.0$) down to absolute zero ($T=0.0$).

* **Hexagonal Crystallization:** As the system cools, particles fall into deep Lennard-Jones potential wells, snapping into a highly ordered 2D hexagonal lattice. The local coordination density histogram proves this with a massive peak at $n=6$.
* **Latent Heat Extraction:** The condensation process generates massive amounts of kinetic energy (heat). To prevent the system from artificially reheating, an aggressive Andersen thermostat ($\nu = 1.0$) was implemented to efficiently scrub the latent heat and force the system to follow the target temperature trajectory.

##  The Polycrystalline State

Instead of forming a single, perfect monocrystal, the rapid cooling schedule forces the fluid to form thousands of independent nucleation sites simultaneously. 

By tracking the **Structural Order Parameter** ($m$)—defined as the fraction of the total particles residing in the largest single connected droplet—we observe that $m$ peaks at less than $5\%$. The resulting solid is a "shattered" polycrystalline state made up of distinct crystal grains with mismatched lattice alignments.

##  Repository Files (Git LFS)

Due to the massive scale of the data, the outputs of this simulation are tracked using **Git LFS**.

* `part2_generate_gpu.py`: The CUDA-accelerated, VRAM-chunked PyTorch simulation script.
* `part2_visualization.py`: The Matplotlib script used to generate the high-resolution dashboard.
* `part2_ladder_massive.npz`: The compressed 113 MB trajectory and thermodynamic data array.
* `massive_32k_dashboard.mp4`: A 397 MB, hardware-accelerated 4K video render of the macroscopic phase transition.

---









