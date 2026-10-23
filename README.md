# Caloris – Lumped Thermal Network Solver

Caloris is an object-oriented Python package for lumped thermal network modeling, supporting decoupled steady-state and transient thermal simulations. 

> **Core Philosophy:** Define the topology using `Node` and `Connection` objects, assemble the graph in a `Network`, and pass the network to standalone numerical `Solver` engines.

## System Architecture

```text
  User Input Model
         │
         ▼
  [Nodes] + [Connections]  ◄── Unidirectional/Bidirectional Graph Elements
         │
         ▼
     [Network]             ◄── Topology Assembly (Conductance G, Fluxes)
         │
         ▼
  [Solver Engines]         ◄── Decoupled Numerical Execution Layers
   ├── SteadySolver
   └── TransientSolver
         │
         ▼
  [Result Objects]         ◄── Structured Data Arrays & Dict Outputs
```

---

## Core Class Reference

### 1. `Node`
Represents a lumped thermal mass or physical boundary. Custom subclasses like `Thermostat` or `Heater` are obsolete; all behaviors are handled via unified boundary configurations.

```python
from Caloris.nodes import Node

node = Node(
    label="N1",                   # Unique string identifier
    initial_temperature=293.15,   # Initial temperature [K]
    mass=0.5,                     # Mass [kg] (for transient cases)
    specific_heat="Al6061",       # Material ID matching material dispatchers
    boundary_type="heat_input",   # Optional: 'temperature' or 'heat_input'
    boundary_value=100.0,         # Value associated with boundary type (W or K)
    boundary_function=None        # Optional callback: lambda T, Q_in: custom_law
)
```

### 2. `Connection`
Represents a thermal coupling between two nodes. Enforces physical symmetry: $G_{ij} = G_{ji}$.

```python
from Caloris.connections import Connection

conn = Connection(
    node_i=node1, 
    node_j=node2, 
    connection_type="conduction", # Options: 'conduction', 'conductance', 'convection', 'radiation'
    A=0.01,                       # Cross-sectional area [m²]
    L=0.1,                        # Path length [m]
    material_conductivity="Al6061",
    # Radiative parameters (Required only if connection_type="radiation")
    e_i=0.8, e_j=0.8, S_i=1.0, S_j=1.0, F_ij=1.0 
)
```

### 3. `Network`
Manages network topology and structural assembly. **It does not perform numerical solving.**

```python
from Caloris.network import Network

net = Network(nodes=[n1, n2], connections=[c1], spread=1.0)

# Key Internal Methods:
# net.validate() -> Checks for duplicate labels and orphan connections
# G = net.build_G(T_array) -> Assembles NxN structural conductance matrix
# fluxes = net.compute_fluxes(T_array) -> Returns dict keyed by 3-tuple: (node_i, node_j, type)
```

### 4. Solvers
Numerical engines are fully decoupled from the network topology.

#### `SteadySolver`
Solves non-linear algebraic system equations using Picard loops or optimization roots.
```python
from Caloris.steady_solver import SteadySolver

solver = SteadySolver(net)
res = solver.solve(verbose=False)

T = res.T          # Flat array of final temperatures matching node indices
fluxes = res.fluxes# Dict of solved branch fluxes. Key format: ('node_i', 'node_j', 'type')
```

#### `TransientSolver`
Performs time-marching implicit numerical integration.
```python
from Caloris.transient_solver import TransientSolver

solver = TransientSolver(net)
result = solver.solve(t_max=3600.0, dt=10.0, verbose=False)

time_steps = result.time         # 1D array of shape (n_steps,)
T_history = result.T_history     # 2D array of shape (n_steps, N_nodes)
```

---

## Complete Blueprint Code Examples

### Example 1: Steady-State Radiative Enclosure
```python
import numpy as np
from Caloris.nodes import Node
from Caloris.connections import Connection
from Caloris.network import Network
from Caloris.steady_solver import SteadySolver

# Setup unified boundary nodes
nodes = [
    Node(label="Plate1", initial_temperature=300.0, boundary_type="temperature", boundary_value=300.0),
    Node(label="Shield", initial_temperature=200.0), # Floating un-bounded node
    Node(label="Space",  initial_temperature=50.0,  boundary_type="temperature", boundary_value=50.0)
]

# Link via radiative couplings
connections = [
    Connection(nodes[0], nodes[1], connection_type="radiation", e_i=0.8, e_j=0.8, S_i=1.0, S_j=1.0, F_ij=1.0),
    Connection(nodes[1], nodes[2], connection_type="radiation", e_i=0.8, e_j=0.8, S_i=1.0, S_j=1.0, F_ij=1.0)
]

net = Network(nodes, connections)
res = SteadySolver(net).solve()

# Structured lookups
flux_key = (nodes[0].label, nodes[1].label, 'radiation')
print(f"Shield Temp: {res.T[1]} K | Primary Flux: {res.fluxes[flux_key]} W")
```

### Example 2: Transient Thermal Mass Run
```python
from Caloris.nodes import Node
from Caloris.connections import Connection
from Caloris.network import Network
from Caloris.transient_solver import TransientSolver

nodes = [
    Node(label="Source", initial_temperature=293.15, boundary_type="heat_input", boundary_value=50.0, mass=1.0, specific_heat="Al6061"),
    Node(label="Sink",   initial_temperature=293.15, boundary_type="temperature", boundary_value=293.15, mass=1.0, specific_heat="Al6061")
]

connections = [
    Connection(nodes[0], nodes[1], connection_type="conduction", A=0.005, L=0.05, material_conductivity="Al6061")
]

net = Network(nodes, connections)
result = TransientSolver(net).solve(t_max=500.0, dt=2.0)

# Data processing layout: rows = timesteps, columns = nodes
final_source_temp = result.T_history[-1, 0] 
```

---

## Dependencies & Setup
```bash
pip install numpy scipy matplotlib
```
* **Supported built-in material properties:** `'Al6061'`, `'SST304L'` (configured inside `Caloris.materials`).
* **Symmetry Tracking:** Branch flux dictionary lookups are sensitive to direction and connection types; use the formal 3-tuple system identifier: `(node_i_label, node_j_label, connection_type)`.
