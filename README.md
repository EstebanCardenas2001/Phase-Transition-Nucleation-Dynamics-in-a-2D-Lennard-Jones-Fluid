# Phase Transition & Nucleation Dynamics in a 2D Lennard-Jones Fluid

<div align="center">
  <img src="dashboard_preview.gif" alt="MD Phase Transition Simulation" width="100%">
=======
  <img src="outputs/dashboard_preview.gif" alt="MD Phase Transition Preview" width="80%">
  
  <br><br>
  
  <p><b> Watch the full macroscopic 4K simulation (Phase 2):</b></p>
  <a href="https://youtu.be/kXBRlVEVq5Y">
    <img src="https://github.com/user-attachments/assets/66cd0fa3-4698-46c1-9743-0584b61f54bc" alt="Macroscopic 32k Simulation Phase Transition" width="80%">
  </a>
</div>

---

##  Project Overview

This repository explores the non-equilibrium thermodynamics of a two-dimensional Lennard-Jones fluid undergoing a first-order phase transition. By applying a continuous cooling ladder, the system spontaneously breaks symmetry, demonstrating droplet nucleation, gas-liquid phase coexistence, and ultimately, crystallization. 

The project is engineered in two distinct phases: 
1. **Local Prototyping:** Developing a robust, fully vectorized custom Molecular Dynamics (MD) engine from scratch.
2. **Cloud HPC Scaling:** Migrating the architecture to GPU-accelerated cloud infrastructure to approach the macroscopic thermodynamic limit.

---

##  Phase 1: Local Prototyping & Physics Engine

The foundational phase focused on algorithm design and building an independent MD engine without relying on external simulation packages (e.g., LAMMPS). The architecture is optimized to maximize single-node CPU throughput using strict matrix vectorization.

* **Interaction Potential:** Standard Lennard-Jones 12-6 potential $V(r) = 4\epsilon \left[ \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6 \right]$ evaluated under periodic boundary conditions (PBC).
* **Vectorized Force Matrix:** The $O(N^2)$ pairwise distance calculations are entirely vectorized via NumPy, leveraging hardware-accelerated linear algebra frameworks to evaluate tens of millions of interactions per integration step.
* **Thermodynamics:** The system state is advanced using a Velocity Verlet integrator coupled with an Andersen thermostat. The protocol involves a strict burn-in phase to thermalize the initial randomized lattice, followed by a stepwise production cooling schedule to induce nucleation.
* **Order Parameter Tracking:** Real-time structural analysis is achieved by computing the local coordination number via a sparse adjacency matrix. This allows for the identification of connected components and the extraction of the structural order parameter ($m$), defined as the largest droplet fraction.

### Analytical Visualization Dashboard

To extract physical meaning from the raw multidimensional trajectory data, a custom 5-panel Matplotlib diagnostic dashboard renders the structural dynamics synchronously with thermodynamic observables:

| Panel | Physical Insight |
| :--- | :--- |
| **Global Domain** | Visualizes the entire periodic simulation box. Particles are continuously color-mapped by their local coordination density. |
| **Magnified Core** | Utilizes a dynamic bounding-box algorithm to anchor the camera to the densest cluster, tracking local droplet condensation in high resolution. |
| **Bimodal Density** | A real-time histogram demonstrating phase coexistence. During cooling, the unified peak bifurcates into distinct low-density (gas) and high-density (liquid/solid) populations. |
| **Energy & Temp** | Maps the actual system temperature against the target thermostat parameter, capturing the steep drops in potential energy as intermolecular bonds form. |
| **Order Parameter** | Plots the largest contiguous cluster fraction ($m$), precisely capturing the critical moment of symmetry breaking. |

## Phase 2: Macroscopic Quench and Phase Coexistence
This phase scales the custom PyTorch molecular dynamics engine to a macroscopic domain ($N=32,768$ particles) to simulate large-scale non-equilibrium phase separation. By leveraging GPU tensor operations and VRAM chunking on the CloudVeneto infrastructure, the engine successfully computed over a billion pairwise interactions per integration step.

## Thermodynamic Quench Dynamics
The system was subjected to a rapid quench using a vectorized Andersen thermostat, driving the fluid out of a homogeneous gas state and deep into the liquid-gas coexistence region.

*   **Supercooling and Thermal Lag:** Because the thermal quench was highly dynamic, the system experienced significant supercooling. It maintained a metastable gaseous state well below the equilibrium boundary, only breaking symmetry when the temperature reached $T \approx 0.30$.
*   **Symmetry Breaking and Nucleation:** The exact moment of condensation was captured quantitatively. A sudden, sharp spike in the macroscopic structural order parameter precisely tracked the spontaneous formation of structured liquid clusters.
*   **Latent Heat Release:** The nucleation event triggered a massive release of latent heat, which was mathematically recorded as a steep, violent drop in the potential energy time-series. 
*   **Domain Formation:** Spatial mapping of the coordinates visually confirmed the phase separation, revealing dense, highly coordinated liquid droplets suspended within a dilute vapor background, perfectly illustrating the binodal coexistence regime.
---

## Phase 3: Critical Point Isolation and Finite-Size Scaling
This phase investigates the thermodynamic limits of the 2D Lennard-Jones fluid, attempting to isolate the liquid-gas critical point and map its universality class. Using a custom batched PyTorch tensor architecture, the system was simulated across multiple finite box sizes ($N \in \{1024, 4096, 16384\}$) to extract macroscopic critical exponents.

## Mathematical Framework
To test the hypothesis that the critical point belongs to the 2D Ising universality class, we analyze the specific heat capacity at constant volume, $C_v$. In the canonical (NVT) ensemble, $C_v$ is derived from the variance of the potential energy $E$:

$$C_v = \frac{N(\langle E^2 \rangle - \langle E \rangle^2)}{T^2}$$

For the 2D Ising model, the critical exponent $\alpha = 0$, dictating that the specific heat diverges logarithmically with the physical box length $L$:

$$C_v \propto \ln(L)$$

Finite-Size Scaling (FSS) dictates that if the system belongs to this exact universality class, plotting the scaled specific heat against the scaled temperature will collapse all discrete system sizes onto a universal master curve:
*   **Scaled Temperature:** $(T - T_c)L^{1/\nu}$ (where $\nu = 1$)
*   **Scaled Specific Heat:** $\frac{C_v}{\ln(L)}$

## Results and Diagnostic Analysis
The execution of the FSS collapse yielded a definitive negative result, providing rigorous mathematical proof of the system's underlying phase behavior at the simulated coordinates ($T_c \approx 0.335$, $\rho_c = 0.329$).

*   **Finite-Size Artifacts:** The bounded $N=1024$ system exhibited a continuous peak, artificially masking the true nature of the transition because the physical domain was too small to support a complete liquid-gas interface.
*   **Rejection of Second-Order Scaling:** The failure of the logarithmic transformation to equalize the peak heights across macroscopic domains ($N=4096$ and $N=16384$) mathematically rejects the continuous 2D Ising hypothesis for this parameter window.
*   **Proof of First-Order Coexistence:** The specific heat variance was observed to scale linearly with the system area ($L^2$). This volumetric scaling is the absolute thermodynamic signature of latent heat release.
*   **Conclusion:** The simulation successfully bypassed the critical point and drove directly through the binodal coexistence line, resulting in a first-order phase transition characterized by macroscopic droplet nucleation.
=======
---

##  Phase 2: Macroscopic Phase Transition

The second phase transitions from studying isolated microscopic droplet formation to simulating a massive, interconnected thermodynamic ensemble. The system is scaled to macroscopic dimensions ($N = 32,768$ particles) to observe latent heat extraction and the nucleation of a polycrystalline solid.

### Computational Physics: The GPU Force Engine

Scaling to 32,768 particles introduces a severe $O(N^2)$ computational bottleneck. Storing the complete periodic distance matrix requires over 14 GB of instantaneous memory, exceeding standard hardware limits and triggering memory fragmentation. 

To overcome this and achieve maximum compute saturation on NVIDIA GPU architecture, the force engine (`part2_generate_gpu.py`) was rebuilt in PyTorch using a **Chunked VRAM Algorithm**:
* **Memory Optimization:** Slices the particle tensor into discrete evaluation chunks, constraining the peak VRAM footprint safely below 3 GB.
* **In-Place Operations:** Executes periodic boundary vector subtractions directly in memory via `.sub_()`, eliminating temporary tensor bloat.
* **Hardware Saturation:** Feeds the CUDA multiprocessors exactly the maximum volume of mathematical operations they can process per cycle, sustaining a continuous 100% GPU utilization rate without bottlenecking the memory bus.

### Thermodynamics & Latent Heat

The simulation evaluates 60,000 integration steps ($10,000$ thermalization, $50,000$ production cooling), driving the system from a high-energy gas ($T=2.0$) to a deep freeze ($T=0.0$).

* **Hexagonal Crystallization:** Particles fall into deep Lennard-Jones potential wells and organize into a highly ordered 2D hexagonal lattice, confirmed by an overwhelming peak at $n=6$ in the local coordination density distribution.
* **Latent Heat Extraction:** Rapid condensation releases immense kinetic energy. An aggressive Andersen thermostat collision frequency ($\nu = 1.0$) was required to efficiently scrub this latent heat and bind the actual temperature to the target trajectory.
* **The Polycrystalline State:** The rapid thermal quench initiates thousands of independent nucleation sites simultaneously. The order parameter ($m$) peaks below $5\%$, indicating the formation of a "shattered" polycrystalline solid comprised of distinct, misaligned crystal grains rather than a single monocrystal.

---

##  Repository Structure & Git LFS

Due to the massive data scale, Phase 2 outputs are version-controlled via **Git Large File Storage (LFS)**.

* `part2_generate_gpu.py`: The CUDA-accelerated, chunked PyTorch force engine.
* `part2_visualization.py`: The high-resolution Matplotlib dashboard rendering script.
* `part2_ladder_massive.npz`: Compressed 113 MB trajectory and thermodynamic coordinate array.
* `massive_32k_dashboard.mp4`: A 397 MB, hardware-accelerated 4K render of the macroscopic transition.
