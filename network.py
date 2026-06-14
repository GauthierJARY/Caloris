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
import matplotlib.pyplot as plt
# import networkx as nx
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

class Network:

    def __init__(self, nodes, connections, spread=1.0):

        self.nodes = nodes
        self.connections = connections
        self.spread = spread

        self.N = len(nodes)

        self.node_to_idx = {
            node: idx
            for idx, node in enumerate(nodes)
        }

        self.idx_to_node = {
            idx: node
            for idx, node in enumerate(nodes)
        }

        self.validate()

    # ==========================================================
    # Validation
    # ==========================================================

    def validate(self):

        labels = [node.label for node in self.nodes]

        if len(labels) != len(set(labels)):
            raise ValueError(
                "Duplicate node labels detected."
            )

        all_nodes = set(self.nodes)

        for conn in self.connections:

            if conn.node_i not in all_nodes:
                raise ValueError(
                    f"{conn.node_i.label} not found in network."
                )

            if conn.node_j not in all_nodes:
                raise ValueError(
                    f"{conn.node_j.label} not found in network."
                )

    # ==========================================================
    # Conductance matrix assembly
    # ==========================================================

    def build_G(self, T):

        G = np.zeros((self.N, self.N))

        for conn in self.connections:

            i = self.node_to_idx[conn.node_i]
            j = self.node_to_idx[conn.node_j]

            G_ij = conn.compute_G(
                T[i],
                T[j],
                spread=self.spread
            )

            G[i, i] += G_ij
            G[j, j] += G_ij

            G[i, j] -= G_ij
            G[j, i] -= G_ij

        return G

    # ==========================================================
    # Heat fluxes
    # ==========================================================

    def compute_fluxes(self, T):

        fluxes = {}

        for conn in self.connections:

            i = self.node_to_idx[conn.node_i]
            j = self.node_to_idx[conn.node_j]

            G_ij = conn.compute_G(
                T[i],
                T[j],
                spread=self.spread
            )

            flux = G_ij * (T[i] - T[j])

            fluxes[
                (
                    conn.node_i.label,
                    conn.node_j.label,
                    conn.connection_type
                )
            ] = flux

        return fluxes

    # def plot(self, save_path="thermal_network.png", figsize=(10, 6), node_size=900, font_size=9):
    #     """
    #     Visualizes the structural topology of the thermal network.
    #     Nodes are color-coded by boundary condition types, and links are colored by physical coupling mechanisms.
    #     """
    #     # 1. Initialize empty networkx Graph object
    #     G = nx.Graph()
        
    #     # 2. Add nodes and capture their boundary condition profiles
    #     for node in self.nodes:
    #         label_text = node.label
    #         b_type = getattr(node, 'boundary_type', None)
    #         if b_type:
    #             label_text += f"\n({b_type})"
    #         G.add_node(node.label, display_label=label_text, boundary_type=b_type)
            
    #     # 3. Add topology edges mapping physical connections
    #     for conn in self.connections:
    #         c_type = getattr(conn, 'connection_type', 'conduction')
    #         G.add_edge(conn.node_i.label, conn.node_j.label, connection_type=c_type)
            
    #     # 4. Set aesthetic color pallets
    #     edge_colors_map = {
    #         'conduction': '#d95f02',   # Terracotta orange
    #         'conductance': '#e7298a',  # Vibrant pink
    #         'convection': '#1b7837',   # Forest green
    #         'radiation': '#7570b3'     # Slate purple
    #     }
        
    #     node_colors_map = {
    #         'temperature': '#fc8d59',  # Light red/salmon for fixed temperature
    #         'heat_input': '#fee08b',   # Yellow/gold for active heaters
    #         None: '#e0f3f8'            # Soft blue for un-bounded floating nodes
    #     }
        
    #     # 5. Compute network node spacing layouts using spring forces
    #     pos = nx.spring_layout(G, seed=42, k=1.5 / np.sqrt(len(G.nodes)))
        
    #     # Gather assigned node colors
    #     node_colors = [node_colors_map.get(G.nodes[n]['boundary_type'], '#e0f3f8') for n in G.nodes]
        
    #     # 6. Build the canvas and render components
    #     fig, ax = plt.subplots(figsize=figsize)
        
    #     # Draw physical node bodies
    #     nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors, edgecolors='black', ax=ax)
        
    #     # Draw string identifiers inside/above nodes
    #     labels = nx.get_node_attributes(G, 'display_label')
    #     nx.draw_networkx_labels(G, pos, labels=labels, font_size=font_size, font_weight='bold', ax=ax)
        
    #     # Draw physical paths grouped by type to neatly organize the legend labels
    #     unique_edge_types = set(nx.get_edge_attributes(G, 'connection_type').values())
    #     for etype in unique_edge_types:
    #         edgelist = [(u, v) for u, v, d in G.edges(data=True) if d['connection_type'] == etype]
    #         color = edge_colors_map.get(etype, '#969696')
    #         nx.draw_networkx_edges(G, pos, edgelist=edgelist, width=2.5, edge_color=color, ax=ax, label=etype)
            
    #     # 7. Construct contextual legend maps
    #     legend_elements = [
    #         Patch(facecolor='#e0f3f8', edgecolor='black', label='Floating Node'),
    #         Patch(facecolor='#fc8d59', edgecolor='black', label='Fixed Temp BC'),
    #         Patch(facecolor='#fee08b', edgecolor='black', label='Heat Input BC'),
    #     ]
    #     for etype in unique_edge_types:
    #         color = edge_colors_map.get(etype, '#969696')
    #         legend_elements.append(Line2D([0], [0], color=color, lw=2.5, label=f"{etype.capitalize()} Link"))
            
    #     ax.legend(handles=legend_elements, loc='upper right', frameon=True, shadow=True)
    #     ax.set_title("Caloris Thermal Network Topology Visualization", fontsize=12, fontweight='bold')
    #     ax.axis('off')
        
    #     # 8. Save output plot
    #     plt.savefig(save_path, bbox_inches='tight', dpi=150)
    #     plt.close()