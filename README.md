# Phase Transition & Nucleation Dynamics in a 2D Lennard-Jones Fluid


<div align="center">
  <img src="outputs/dashboard_preview.gif" alt="MD Phase Transition Simulation" width="100%">
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

## Phase 2: Cloud Scaling (In Progress)

While pure NumPy vectorization handles $N = 4,000$ to $N = 8,000$ efficiently for overnight execution, approaching the true thermodynamic limit requires migrating the $O(N^2)$ tensor algebra to dedicated hardware. 

The next phase of this project scales the simulation environment to a **Tesla T4 GPU instance on CloudVeneto**. By refactoring the compute core into PyTorch, the matrix operations will run entirely in VRAM via CUDA, allowing massive system sizes ($N > 20,000$) and highly granular temperature ladders in a fraction of the compute time.

---









