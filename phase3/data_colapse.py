import numpy as np
import matplotlib.pyplot as plt

# --- 1. System Parameters ---
files = ['fss_final_N1024.npz', 'fss_final_N4096.npz', 'fss_final_N16384.npz']
colors = ['#1f77b4', '#ff7f0e', '#d62728']

# 2D Ising Critical Exponents
nu_ising = 1.0 
alpha_ising = 0.0 # Logarithmic scaling expectation

# Critical temperature identified from N=1024 verification
Tc = 0.335

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

print("Executing Finite-Size Scaling Collapse...")

for idx, file_name in enumerate(files):
    # --- 2. Data Extraction ---
    try:
        data = np.load(file_name)
    except FileNotFoundError:
        print(f"Error: {file_name} not found in the current directory.")
        continue
        
    T = data['T_target']
    E_series = data['E_series']
    N = int(data['N'])
    L = float(data['L'])
    
    # --- 3. Fluctuation Thermodynamics ---
    # Calculate variance along the time axis (axis=0)
    E_var = np.var(E_series, axis=0)
    
    # Specific heat capacity per particle
    Cv = (E_var * N) / (T**2)
    
    # Panel 1: Raw Macroscopic Divergence
    ax1.plot(T, Cv, marker='o', color=colors[idx], linewidth=2, label=f'N={N}')
    
    # --- 4. Universality Scaling Laws ---
    # Scaled Temperature: (T - Tc) * L^(1/nu)
    x_scaled = (T - Tc) * (L ** (1 / nu_ising))
    
    # Scaled Specific Heat: Cv / ln(L) due to alpha = 0
    y_scaled = Cv / np.log(L)
    
    # Panel 2: FSS Data Collapse
    ax2.plot(x_scaled, y_scaled, marker='s', color=colors[idx], linewidth=2, label=f'L={L:.1f}')

# --- 5. Figure Formatting ---
ax1.set_title('Divergence of Specific Heat Capacity')
ax1.set_xlabel('Temperature ($T$)')
ax1.set_ylabel(r'$C_v / N$')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

ax2.set_title('Universality Collapse (2D Ising)')
ax2.set_xlabel(r'Scaled Temperature $(T - T_c)L^{1/\nu}$')
ax2.set_ylabel(r'Scaled Specific Heat $C_v / \ln(L)$')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('final_ising_collapse.png', dpi=300)
print("Processing complete. Figure exported as 'final_ising_collapse.png'.")
plt.show()