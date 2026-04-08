# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 18:47:50 2025

@author: G.J.
"""
# -*- coding: utf-8 -*-
"""
Object-oriented Network class for steady-state thermal solving.
Faithfully reproduces old procedural solver logic.
"""

import numpy as np
from Caloris.nodes import Thermostat, Node, Cryostat, Heater
from Caloris.materials import cp_material_dispatch
from scipy.optimize import root
from scipy.integrate import solve_ivp

class Network:
    def __init__(self, nodes, connections, spread=1.0):
        self.nodes = nodes
        self.connections = connections
        self.node_to_idx = {node.label: idx for idx, node in enumerate(self.nodes)}
        self.idx_to_node = {idx: node for idx, node in enumerate(self.nodes)}
        self.N = len(nodes)
        self.spread = spread
        self._validate_network()
        
    # -------------------------------------------------------------------------
    def _validate_network(self):
        """Validate the network structure."""
        all_node_labels = {node.label for node in self.nodes}
        for conn in self.connections:
            if (conn.node_i.label not in all_node_labels or
                conn.node_j.label not in all_node_labels):
                raise ValueError("Connection references non-existent node")
    # -------------------------------------------------------------------------  
    def _get_node_temperature(self, label):
        """Get temperature of node by label."""
        return self.nodes[self.node_to_idx[label]].temperature
    # -------------------------------------------------------------------------
    def _set_node_temperature(self, label, temperature):
        """Set temperature of node by label."""
        self.nodes[self.node_to_idx[label]].temperature = temperature

    # -------------------------------------------------------------------------
    def build_G(self, T):
        """Build conductance matrix G(T) without mutating nodes."""
        N = self.N
        G = np.zeros((N, N))
    
        for conn in self.connections:
            i = self.node_to_idx[conn.node_i.label]
            j = self.node_to_idx[conn.node_j.label]
    
            T_i = T[i]
            T_j = T[j]
    
            G_ij = conn.compute_G(T_i, T_j, spread=self.spread)
    
            G[i, i] += G_ij
            G[j, j] += G_ij
            G[i, j] -= G_ij
            G[j, i] -= G_ij
    
        return G
    
    # -------------------------------------------------------------------------
    def compute_fluxes(self, T, G):
        """Compute fluxes from G and T."""
        fluxes = {}
    
        for conn in self.connections:
            i = self.node_to_idx[conn.node_i.label]
            j = self.node_to_idx[conn.node_j.label]
    
            G_ij = -G[i, j]  # off-diagonal = -G_ij
            flux = G_ij * (T[i] - T[j])
    
            if flux >= 0:
                fluxes[(conn.node_i.label, conn.node_j.label)] = flux
            else:
                fluxes[(conn.node_j.label, conn.node_i.label)] = -flux
    
        return fluxes

    # -------------------------------------------------------------------------
    def apply_special_boundary_conditions(self, G, S, fluxes, C=None):
        """
        Apply thermostat, cryostat, and heater boundary conditions.
        If C is provided (transient case), modify it consistently for Dirichlet nodes.
        
        Parameters:
        - G: Conductance matrix (NxN)
        - S: Source vector (Nx1)
        - fluxes: dictionary of fluxes
        - C: Optional capacitance matrix (NxN) for transient problems
        """
        for node in self.nodes:
            idx = self.node_to_idx[node.label]
    
            # --- Thermostat: fixed temperature (Dirichlet)
            if isinstance(node, Thermostat):
                G[idx, :] = 0
                G[idx, idx] = 1.0
                S[idx] = node.behaviour  # fixed temperature
                if C is not None:
                    C[idx, :] = 0   # derivative term removed for Dirichlet enforcement
                    C[idx, idx] = 1e-8  # small number to prevent divide-by-zero
            # --- Cryostat: dynamic temperature via behaviour(T, Q)
            elif isinstance(node, Cryostat):
                G[idx, :] = 0
                G[idx, idx] = 1.0
                sum_incoming_fluxes = sum(value for (source, dest), value in fluxes.items() if dest == node.label)
                S[idx] = node.behaviour(node.temperature, sum_incoming_fluxes)
                if C is not None:
                    C[idx, :] = 0   # remove derivative term
                    C[idx, idx] = 1e-8  # small number to prevent divide-by-zero
            # --- Heater: acts as power input
            elif isinstance(node, Heater):
                sum_incoming_fluxes = sum(value for (source, dest), value in fluxes.items() if dest == node.label)
                S[idx] += node.behaviour(node.temperature, sum_incoming_fluxes) if callable(node.behaviour) else node.behaviour
        if C is None :
            return G, S
        else: 
            return G, S, C 

    # -------------------------------------------------------------------------
    def solve_steady(self, tol=1e-5, max_iter=100, verbose=True):
       """Stateless steady-state solver."""
       
       T = np.array([node.temperature for node in self.nodes])
       Q = np.array([node.heat_input for node in self.nodes])
       
       convergence_history = []
       
       for it in range(max_iter):
       
           # Build system
           G = self.build_G(T)
           fluxes = self.compute_fluxes(T, G)
       
           S = Q.copy()
           G_bc, S_bc = self.apply_special_boundary_conditions(G.copy(), S.copy(), fluxes)
       
           # Solve linear system
           try:
               T_new = np.linalg.solve(G_bc, S_bc)
           except np.linalg.LinAlgError:
               raise ValueError("❌ Singular G matrix")
       
           # Physical constraint
           T_new = np.maximum(T_new, 0.0)
       
           # Convergence
           err = np.max(np.abs(T_new - T))
           convergence_history.append(err)
       
           if err < tol:
               if verbose:
                   print(f"✅ Converged in {it+1} iterations.")
               break
       
           # Under-relaxation
           T = 0.5 * T + 0.5 * T_new
       
       # Final recompute (IMPORTANT)
       G = self.build_G(T)
       fluxes = self.compute_fluxes(T, G)
       
       # Store internally (fix #5)
       self._last_G = G
       self._last_fluxes = fluxes
       self._last_T = T
       
       # Return structured result (fix #2)
       return {
           "T": T,
           "fluxes": fluxes,
           "G": G,
           "convergence": convergence_history
       }
   
    # -------------------------------------------------------------------------
    def solve_steady_scipy(self, tol=1e-5, max_iter=150, method='hybr', verbose=True):
        """
        Solve for steady-state temperatures using scipy.optimize.root instead of manual Picard iteration.
        Solves F(T) = 0 where F(T) = G(T)*T - S(T).
        """

        # Initial guess from current node temperatures
        T0 = np.array([node.temperature for node in self.nodes])

        def residual(T):
            """Compute residual vector F(T) = G(T)*T - S(T)."""
            # Update node temperatures
            for i, node in enumerate(self.nodes):
                node.temperature = T[i]

            # Build conductance and flux dictionary
            G, fluxes = self.build_G(T)
            S = np.array([node.heat_input for node in self.nodes])

            # Apply special boundary conditions (Dirichlet, etc.)
            G_bc, S_bc = self.apply_special_boundary_conditions(G.copy(), S.copy(), fluxes)

            # Compute residual F(T)
            F = G_bc @ T - S_bc
            return F

        # Solve nonlinear system using scipy.root
        sol = root(residual, T0, method=method, tol=tol, options={'maxfev': max_iter})

        if not sol.success:
            raise RuntimeError(f"❌ Steady solver failed to converge: {sol.message}")

        T_final = sol.x

        # Update node temperatures
        for i, node in enumerate(self.nodes):
            node.temperature = T_final[i]

        # Recompute final fluxes
        _, fluxes = self.build_G(T)

        if verbose:
            print(f"✅ Steady-state converged: {sol.message}")
            print(f"Iterations: {sol.nfev} | Final max residual: {np.max(np.abs(sol.fun)):.3e}")

        return T_final, fluxes, sol

    
    # -------------------------------------------------------------------------
    def build_C(self):
        """
        Build diagonal matrix C of thermal capacitances for each node.
        Care when manipulating c_p or C_p and addition of the mass or volumic mass rho !
        """
        C_diag = np.array([node.mass * cp_material_dispatch(node.temperature, node.material_specific_heat) for node in self.nodes])
        return np.diag(C_diag) # should add the spreading for uncertainty

    # -------------------------------------------------------------------------
    def solve_transient(self, t_max, dt=1e-3, T_init=None, verbose=True):
        """
        Solve the transient heat transfer problem using implicit Euler method.
        
        Parameters:
        - t_max: total simulation time
        - dt: time step size
        - T_init: initial temperature array (optional)
        - verbose: whether to print time-step progress
        
        Returns:
        - T_hist: list of temperature arrays at each time step
        - time_points: list of time points
        """
        
        n_steps = int(t_max / dt)
        T = np.array([node.temperature for node in self.nodes]) if T_init is None else T_init.copy()
        Q = np.array([node.heat_input for node in self.nodes])
    
        T_hist = [T.copy()]
        time_points = [0.0]
    
        for step in range(1, n_steps + 1):
            t = step * dt
    
            # Fixed-point (Picard) iteration for nonlinear solve, simple but less robust, one may consider to implement NEwton Raphson method instead
            for _ in range(20):  # max inner iterations
                # Update node temperatures (required if G, C depend on T)
                for i, node in enumerate(self.nodes):
                    node.temperature = T[i]
    
                G = self.build_G(T)
                fluxes = self.compute_fluxes(T, G)
                C = self.build_C()
                S = Q.copy()
    
                # Build system: (C/dt + G) T_new = C*T/dt + S
                # Apply special BCs (thermostats, cryostats, heaters, etc.) to A b and no longer to G S 
                # Apply BCs correctly
                G_bc, S_bc, C_bc = self.apply_special_boundary_conditions(G.copy(), S.copy(), fluxes, C=C)
                
                # Build system: (C/dt + G) T_new = C*T/dt + S
                A = C_bc / dt + G_bc
                b = C_bc @ (T / dt) + S_bc
                try:
                    T_new = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    raise ValueError("❌ Singular matrix in transient solver")
    
                # Enforce physical realism
                T_new = np.maximum(T_new, 0.0)
    
                # Check convergence of nonlinear solve
                err = np.max(np.abs(T_new - T))
                if err < 1e-5:
                    break
                # Optionally do under-relaxation if needed
                T = 0.5 * T + 0.5 * T_new
    
            # Update current temperature
            T = T_new
            T_hist.append(T.copy())
            time_points.append(t)
    
            if verbose and step % 10 == 0:
                print(f"Step {step}/{n_steps} | Time = {t:.2f} | max(T) = {T.max():.2f}")
    
        return np.array(T_hist), np.array(time_points)

    
    # -----------------------------
    # rhs_transient with correct BC enforcement
    # -----------------------------
    def rhs_transient(self, t, T):
        """
        Right-hand side function for transient system: dT/dt = f(T, t)
        """
        # Update node temperatures
        for i, node in enumerate(self.nodes):
            node.temperature = T[i]
    
        # Build matrices
        G = self.build_G(T)
        fluxes = self.compute_fluxes(T, G)  # compute fluxes separately
        C = self.build_C()
        S = np.array([node.heat_input for node in self.nodes])
    
        # Apply boundary conditions (pass C to handle Dirichlet)
        G, S, C = self.apply_special_boundary_conditions(G.copy(), S.copy(), fluxes, C=C)
    
        # Invert C (diagonal)
        C_inv = np.diag(1.0 / np.diag(C))
        
        # Compute dT/dt
        dTdt = C_inv @ (-G @ T + S)
        return dTdt


    # -------------------------------------------------------------------------
    # -----------------------------
    # solve_ivp_transient uses rhs_transient
    # -----------------------------
    def solve_ivp_transient(self, t_span, dt=1.0, T_init=None, method='BDF', rtol=1e-6, atol=1e-9, verbose=True):
        if T_init is None:
            T_init = np.array([node.temperature for node in self.nodes])
            
        t_eval = np.arange(t_span[0], t_span[1] + dt/2, dt)

        sol = solve_ivp(
            fun=self.rhs_transient,
            t_span=t_span,
            y0=T_init,
            method=method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol
        )
    
        # Update final temperatures
        for i, node in enumerate(self.nodes):
            node.temperature = sol.y[i, -1]
    
        if verbose:
            print(f"✅ Transient solve complete: {len(sol.t)} time points, final T_mean = {np.mean(sol.y[:, -1]):.2f} K")
    
        return sol







