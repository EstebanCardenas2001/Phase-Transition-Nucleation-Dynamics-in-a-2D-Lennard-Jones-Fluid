# Phase Transition & Nucleation Dynamics in a 2D Lennard-Jones Fluid


<div align="center">
  <img src="dashboard_preview.gif" alt="MD Phase Transition Simulation" width="100%">
</div>

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







