# Caloris or the FDM-thermal-solver

**Caloris** is a Python package for **lumped thermal network modeling**, supporting both **steady-state** and **transient simulations** of nodes connected by conduction, contact, or radiative links. It allows for **custom boundary conditions** (thermostats, cryostats, heaters), **temperature-dependent materials**, and flexible network configurations.  
It aims at quick estimations and grasp physics within preliminary thermal design of projects. More will be done to make it more intuitive. 

- you can compare the examples with : https://ntrs.nasa.gov/api/citations/20200006182/downloads/Introduction%20to%20Numerical%20Methods%20in%20Heat%20Transfer.pdf where we find the same results as the 2 developped examples for transient conduction. 
## How implement your model correctly ? 
Only put one connection link, it handles the symmetry correctly, even for radiation. Be careful with view factor for radiation. 

---

## Features

- **Node-based thermal modeling**  
  - Supports standard nodes, heaters, thermostats, and cryostats  
  - Node properties: temperature, mass, material, heat input  

- **Connection types**  
  - Conduction (`conduction`)  
  - Contact (`contact`)  
  - Radiation (`radiation`)  
  - Directly specified conductance (`direct_G`)  

- **Steady-state solver**  
  - Picard iteration (`solve_steady`)  
  - Scipy-based root finding (`solve_steady_scipy`)  

- **Transient solver**  
  - Implicit Euler / fixed-point iteration (`solve_transient`)  
  - Scipy IVP solver for stiff systems (`solve_ivp_transient`)  

- **Temperature-dependent material properties**  
  - Thermal conductivity and specific heat (e.g., Al6061, SST304L)  

- **Mass and thermal capacitance calculation**  
  - Automatically computes node masses from adjacent links  
  - Thermal capacitance: \(C_i = m_i \cdot c_p(T_i)\)  

- **Visualization utilities**  
  - Temperature vs time plots  
  - Temperature vs position plots  

---

## Installation

Clone the repository and add it to your Python path:

```bash
git clone https://github.com/yourusername/Caloris.git
cd Caloris
